# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import copy
import logging
import os.path
import shutil, os
import traceback
from typing import Optional
from scipy.constants import mu_0

from scipy.interpolate import interp1d
from .._core.util import Struct
import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    import spec.spec_f90wrapped as spec
except ImportError as e:
    spec = None
    logger.debug(str(e))

try:
    import py_spec
except ImportError as e:
    py_spec = None
    logger.debug(str(e))

try:
    import pyoculus
except ImportError as e:
    pyoculus = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure
from ..field.normal_field import NormalField
from ..geo.surfacerzfourier import SurfaceRZFourier
from ..geo import Surface
from .profiles import ProfileSpec

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None


__all__ = ['Spec', 'Residue', 'QuasisymmetryRatioResidualSpec']


class Spec(Optimizable):
    """
    This class represents the SPEC equilibrium code.

    Philosophy regarding mpol and ntor: The Spec object keeps track of
    mpol and ntor values that are independent of those for the
    boundary Surface object. If the Surface object has different
    mpol/ntor values, the Surface's rbc/zbs arrays are first copied,
    then truncated or expanded to fit the mpol/ntor values of the Spec
    object to before Spec is run. Therefore, you may sometimes need to
    manually change the mpol and ntor values for the Spec object.

    The default behavior is that all  output files will be
    deleted except for the first and most recent iteration on worker
    group 0. If you wish to keep all the output files, you can set
    ``keep_all_files = True``. If you want to save the output files
    for a certain intermediate iteration, you can set the
    ``files_to_delete`` attribute to ``[]`` after that run of SPEC.

    Args:
        filename: SPEC input file to use for initialization. It should end
          in ``.sp``. Or, if None, default values will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for SPEC calculations. If ``None``,
          each MPI process will run SPEC independently.
        verbose: Whether to print SPEC output to stdout (and also a few other statements)
        keep_all_files: If ``False``, all output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all output files will be kept.
        tolerance: Max force balance residue to consider the equilibrium as
          converged; if :math:`|f|>` tolerance, raise ``ObjectiveFailure`` exception. By
          default set to 1E-12.
    """

    def __init__(self,
                 filename: Optional[str] = None,
                 mpi: Optional[MpiPartition] = None,
                 verbose: bool = True,
                 keep_all_files: bool = False,
                 tolerance: float = 1e-12):

        if spec is None:
            raise RuntimeError(
                "Using Spec requires spec python wrapper to be installed.")
        if py_spec is None:
            raise RuntimeError(
                "Using Spec requires py_spec to be installed.")
        if tolerance <= 0:
            raise ValueError(
                'tolerance should be greater than zero'
            )
        self.tolerance = tolerance

        self.lib = spec
        # For the most commonly accessed fortran modules, provide a
        # shorthand:
        self.allglobal = getattr(spec, "allglobal")
        self.inputlist = getattr(spec, "inputlist")
        si = self.inputlist  # Shorthand

        self.verbose = verbose
        # mute screen output if necessary
        # TODO: mute SPEC in windows, currently relies on /dev/null being accessible
        if not self.verbose:
            self.lib.fileunits.mute(1)

        # If mpi is not specified, use a single worker group:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        # SPEC will use the "groups" communicator from the MpiPartition:
        self.allglobal.set_mpi_comm(self.mpi.comm_groups.py2f())

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'defaults.sp')
            if self.mpi.proc0_groups:
                logger.info(
                    f"Initializing a SPEC object from defaults in {filename}")
        else:
            if not filename.endswith('.sp'):
                filename = f"{filename}.sp"
            if self.mpi.proc0_groups:
                logger.info(f"Group {self.mpi.group}: Initializing a SPEC object from file: {filename}")
        
        #If spec has run before, clear the f90wrap array caches.
        # See https://github.com/hiddenSymmetries/simsopt/pull/431
        # This addresses the issue https://github.com/hiddenSymmetries/simsopt/issues/357
        if self.allglobal._arrays:
            self._clear_f90wrap_array_caches()

        # Initialize the FORTRAN state with values in the input file:
        self._init_fortran_state(filename)

        # Copy useful and commonly used values from SPEC inputlist to 
        self.stellsym = bool(si.istellsym)
        self.freebound = bool(si.lfreebound)
        self.nfp = si.nfp
        self.mpol = si.mpol
        self.ntor = si.ntor
        self.nvol = si.nvol
        if self.freebound:
            self.mvol = si.nvol + 1
        else: 
            self.mvol = si.nvol

        # Store initial guess data
        # The initial guess is a collection of SurfaceRZFourier instances,
        # stored in a list of size Mvol-1 (the number of inner interfaces)
        read_initial_guess = self.inputlist.linitialize == 0 and self.nvol > 1 
        if read_initial_guess:
            self.initial_guess = self._read_initial_guess()
            # In general, initial guess is NOT a degree of freedom for the
            # optimization - we thus fix them.
            for surface in self.initial_guess:
                surface.fix_all()
        else: 
            # There is no initial guess - in this case, we let SPEC handle
            # the construction of the initial guess. This generally means
            # that the geometry of the inner interfaces will be constructed
            # by interpolation between the plasma (or computational) boundary
            # and the magnetic axis
            self.initial_guess = None

        # Store axis data
        self.axis = {}
        self.axis['rac'] = copy.copy(si.rac[0:si.ntor+1])
        self.axis['zas'] = copy.copy(si.zas[0:si.ntor+1])
        if si.istellsym == 0:
            self.axis['ras'] = copy.copy(si.ras[0:si.ntor+1])
            self.axis['zac'] = copy.copy(si.zac[0:si.ntor+1])

        self.extension = filename[:-3]
        self.keep_all_files = keep_all_files
        self.files_to_delete = []

        # Create a surface object for the boundary:
        self._boundary = SurfaceRZFourier(nfp=self.nfp, stellsym=self.stellsym,
                                          mpol=self.mpol, ntor=self.ntor)
        self._boundary.rc[:] = self.array_translator(si.rbc, style='spec').as_simsopt
        self._boundary.zs[:] = self.array_translator(si.zbs, style='spec').as_simsopt 
        if not self.stellsym:
            self._boundary.rs[:] = self.array_translator(si.rbs, style='spec').as_simsopt
            self._boundary.zc[:] = self.array_translator(si.zbc, style='spec').as_simsopt
        self._boundary.local_full_x = self._boundary.get_dofs()

        # If the equilibrium is freeboundary, we need to read the computational
        # boundary as well. Otherwise set the outermost boundary as the
        # computational boundary. 
        if self.freebound:
            self._computational_boundary = SurfaceRZFourier(nfp=self.nfp, stellsym=self.stellsym,
                                                            mpol=self.mpol, ntor=self.ntor)
            self._computational_boundary.rc[:] = self.array_translator(si.rwc, style='spec').as_simsopt
            self._computational_boundary.zs[:] = self.array_translator(si.zws, style='spec').as_simsopt 
            if not self.stellsym:
                self._computational_boundary.rs[:] = self.array_translator(si.rws, style='spec').as_simsopt
                self._computational_boundary.zc[:] = self.array_translator(si.zwc, style='spec').as_simsopt
            self._computational_boundary.local_full_x = self._computational_boundary.get_dofs()

        self.need_to_run_code = True
        self.counter = -1

        # Set profiles as None - these have to be defined in a script if the user
        # wish to use them as degrees of freedom
        self._volume_current_profile = None
        self._interface_current_profile = None
        self._pressure_profile = None
        self._iota_profile = None
        self._oita_profile = None
        self._mu_profile = None
        self._pflux_profile = None
        self._tflux_profile = None
        self._helicity_profile = None

        # Define normal field - these are the Vns, Vnc harmonics. Can be used as
        # dofs in an optimization
        if self.freebound:
            vns = self.array_translator(si.vns)
            vnc = self.array_translator(si.vnc)
            self._normal_field = NormalField(nfp=self.nfp, stellsym=self.stellsym,
                                             mpol=self.mpol, ntor=self.ntor,
                                             vns=vns.as_simsopt, vnc=vnc.as_simsopt, 
                                             surface=self._computational_boundary)
        else:
            self._normal_field: Optional[NormalField] = None

        # By default, all dofs owned by SPEC directly, as opposed to
        # dofs owned by the boundary surface object, are fixed.
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge', 'curtor']
        if self.freebound:
            depends_on = [self._normal_field]
        else:
            depends_on = [self._boundary]

        super().__init__(x0=x0, fixed=fixed, names=names,
                         depends_on=depends_on,
                         external_dof_setter=Spec.set_dofs)
        
    @classmethod
    def default_freeboundary(cls, copy_to_pwd, verbose=True):
        """
        Create a default freeboundary SPEC object
        Args:
            copy_to_pwd: boolean, if True, the default input file will be copied to the current working directory. Has to be set True as free-boundary SPEC can only handle files in the current working directory.
            verbose: boolean, if True, print statements will be printed
        """
        filename = 'defaults_freebound.sp'
        if verbose: 
            print(f'Copying {os.path.join(os.path.dirname(__file__), filename)} to {os.getcwd()}/{filename}')
        shutil.copy(os.path.join(os.path.dirname(__file__), filename), os.getcwd()) 
        return cls(filename=filename, verbose=verbose)

    @classmethod
    def default_fixedboundary(cls):
        """
        Create a default fixedboundary SPEC object (same as calling class empty)
        """
        return cls()

    def _read_initial_guess(self):
        """
        Read initial guesses from SPEC and return a list of surfaceRZFourier objects.
        """
        nmodes = self.allglobal.num_modes
        interfaces = []    # initialize list
        n_guess_surfs = self.nvol if self.freebound else self.nvol - 1  # if freeboundary, plasma boundary is also a guess surface
        for lvol in range(0, n_guess_surfs):  # loop over volumes
            # the fourier comonents of the initial guess are stored in the allrzrz array
            thissurf_rc = self.allglobal.allrzrz[0, lvol, :nmodes]
            thissurf_zs = self.allglobal.allrzrz[1, lvol, :nmodes]
            if not self.stellsym:
                thissurf_rs = self.allglobal.allrzrz[2, lvol, :nmodes]
                thissurf_zc = self.allglobal.allrzrz[3, lvol, :nmodes]
            thissurf = SurfaceRZFourier(nfp=self.nfp, stellsym=self.stellsym,
                                        mpol=self.mpol, ntor=self.ntor)
            # the mode number convention is different in SPEC. Best to use the
            # in-built array mmrzrz to get the numbers for each fourier component:
            for imode in range(0, nmodes):
                mm = self.allglobal.mmrzrz[imode]  # helper arrays to get index for each mode
                nn = self.allglobal.nnrzrz[imode]
                # The guess surface could have more modes than we are running SPEC with, skip if case
                if mm > self.mpol or abs(nn) > self.ntor:
                    continue
                thissurf.set_rc(mm, nn, thissurf_rc[imode])
                thissurf.set_zs(mm, nn, thissurf_zs[imode])
                if not self.stellsym:
                    thissurf.set_rs(mm, nn, thissurf_rs[imode])
                    thissurf.set_zc(mm, nn, thissurf_zc[imode])
            interfaces.append(thissurf)
        return interfaces
    
    def _set_spec_initial_guess(self):
        """
        Set initial guesses in SPEC from a list of surfaceRZFourier objects.
        """
        # Set all modes to zero
        self.allglobal.mmrzrz[:] = 0
        self.allglobal.nnrzrz[:] = 0
        self.allglobal.allrzrz[:] = 0

        # transform to SurfaceRZFourier if necessary
        initial_guess = [s.to_RZFourier() for s in self.initial_guess]

        # Loop on modes
        imn = -1  # counter
        for mm in range(0, self.mpol+1):
            for nn in range(-self.ntor, self.ntor+1):
                if mm == 0 and nn < 0:
                    continue

                imn += 1

                self.allglobal.mmrzrz[imn] = mm
                self.allglobal.nnrzrz[imn] = nn

                # Populate inner plasma boundaries
                for lvol in range(0, self.nvol-1):
                    self.allglobal.allrzrz[0, lvol, imn] = initial_guess[lvol].get_rc(mm, nn)
                    self.allglobal.allrzrz[1, lvol, imn] = initial_guess[lvol].get_zs(mm, nn)

                    if not self.stellsym:
                        self.allglobal.allrzrz[2, lvol, imn] = initial_guess[lvol].get_rs(mm, nn)
                        self.allglobal.allrzrz[3, lvol, imn] = initial_guess[lvol].get_zc(mm, nn)
        self.allglobal.num_modes = imn + 1
    # possibly cleaner way to do this (tested to work Chris Smiet 11/9/2023):
#            n_indices, m_values = np.indices([2*si.ntor+1, si.mpol+1]) #get indices for array
#            n_values = n_indices - si.ntor # offset indices to correspond with mode numbers
#            index_mask = ~np.logical_and(m_values==0, n_values < 0) # twiddle ~ NOT, pick elements
#            numel = np.sum(index_mask) # number of trues in mask, i.e. chosen elements
#            self.allglobal.mmrzrz[:numel] = m_values[index_mask] # skip elements, make 1
#            self.allglobal.nnrzrz[:numel] = n_values[index_mask] # skip m==0 n<0 elements
#            
#            initial_guess = [s.to_RZFourier() for s in self.initial_guess]
#            for lvol, surface in enumerate(initial_guess):
#                # EXPLAIN: surface.rc is m,n indexed, transpose, apply above mask which unravels to 1d
#                self.allglobal.allrzrz[0, lvol, :numel] = surface.rc.transpose()[index_mask] 
#                self.allglobal.allrzrz[1, lvol, :numel] = surface.zs.transpose()[index_mask]
#                if not self.stellsym:
#                    self.allglobal.allrzrz[2, lvol, :numel] = surface.rs.transpose()[index_mask]
#                    self.allglobal.allrzrz[3, lvol, :numel] = surface.zc.transpose()[index_mask]

    @property
    def boundary(self):
        """
        Getter for the plasma boundary

        Returns:
            SurfaceRZFourier instance representing the plasma boundary
        """
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        """
        Setter for the geometry of the plasma boundary
        """
        if not self.freebound: 
            if self._boundary is not boundary:
                self.remove_parent(self._boundary)
                self._boundary = boundary
                self.append_parent(boundary)
                return
        else:  # in freeboundary case, plasma boundary is is not a parent
            self._boundary = boundary
    
    @property
    def normal_field(self):
        """
        Getter for the normal field
        """
        return self._normal_field
    
    @normal_field.setter
    def normal_field(self, normal_field):
        """
        Setter for the normal field
        """
        if not self.freebound:
            raise ValueError('Normal field can only be set in freeboundary case')
        if not isinstance(normal_field, NormalField):
            raise ValueError('Input should be a NormalField or CoilNormalField')
        if self._normal_field is not normal_field:
            self.remove_parent(self._normal_field)
            if self._computational_boundary is not normal_field.surface:
                normal_field.surface = self._computational_boundary
            self._normal_field = normal_field
            self.append_parent(normal_field)
            return

    @property
    def computational_boundary(self):
        """
        Getter for the computational boundary. 
        Same as the plasma boundary in the non-free-boundary case, 
        gives the surface on which Vns and Vnc are defined in the free-boundary case.

        Returns:
            SurfaceRZFourier instance representing the plasma boundary
        """
        return self._computational_boundary

    @property
    def pressure_profile(self):
        """
        Getter for the pressure profile

        Returns:
            ProfileSpec instance representing the pressure profile
        """
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
        """
        Setter for the pressure profile

        Args:
            ProfileSpec instance for the pressure profile
        """

        # Check inputs
        if not isinstance(pressure_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if pressure_profile.dofs.full_x.size != self.nvol:
            raise ValueError('Invalid number of dofs. Should be equal to nvol!')

        # Update pressure profile
        if pressure_profile is not self._pressure_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._pressure_profile is not None:
                self.remove_parent(self._pressure_profile)
            self._pressure_profile = pressure_profile
            if pressure_profile is not None:
                self.append_parent(pressure_profile)
                self.need_to_run_code = True

    @property
    def volume_current_profile(self):
        """
        Getter for the volume current profile (Ivolume)

        Returns:
            ProfileSpec instance representing the volume current profile
        """
        return self._volume_current_profile

    @volume_current_profile.setter
    def volume_current_profile(self, volume_current_profile):
        """
        Setter for the volume current profile

        Args:
            ProfileSpec instance for the volume current profile
        """

        if not isinstance(volume_current_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if volume_current_profile.dofs.full_x.size != self.nvol:
            raise ValueError('Invalid number of dofs. Should be equal to nvol!')

        # Volume current is a cumulative property
        volume_current_profile.cumulative = True

        if volume_current_profile is not self._volume_current_profile:
            logging.debug('Replacing volume_current_profile in setter')
            if self._volume_current_profile is not None:
                self.remove_parent(self._volume_current_profile)
            self._volume_current_profile = volume_current_profile
            if volume_current_profile is not None:
                self.append_parent(volume_current_profile)
                self.need_to_run_code = True

    @property
    def interface_current_profile(self):
        """
        Getter for the surface current profile (Isurf)

        Returns:
            ProfileSpec instance representing the surface current profile
        """
        return self._interface_current_profile

    @interface_current_profile.setter
    def interface_current_profile(self, interface_current_profile):
        """
        Setter for the surface current profile

        Args:
            ProfileSpec instance for the surface current profile
        """

        if not isinstance(interface_current_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if interface_current_profile.dofs.full_x.size != self.nvol-1:
            raise ValueError('Invalid number of dofs. Should be equal to nvol-1!')

        if interface_current_profile is not self._interface_current_profile:
            logging.debug('Replacing interface_current_profile in setter')
            if self._interface_current_profile is not None:
                self.remove_parent(self._interface_current_profile)
            self._interface_current_profile = interface_current_profile
            if interface_current_profile is not None:
                self.append_parent(interface_current_profile)
                self.need_to_run_code = True

    @property
    def iota_profile(self):
        """
        Getter for the inner rotational transform profile (iota)

        Returns:
            ProfileSpec instance representing the iota profile
        """
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
        """
        Setter for the inner rotational transform profile (iota)

        Args:
            ProfileSpec instance for the inner rotational transform profile
        """

        if not isinstance(iota_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if iota_profile.dofs.full_x.size != self.nvol+1:
            raise ValueError('Invalid number of dofs. Should be equal to nvol+1!')

        if iota_profile is not self._iota_profile:
            logging.debug('Replacing iota_profile in setter')
            if self._iota_profile is not None:
                self.remove_parent(self._iota_profile)
            self._iota_profile = iota_profile
            if iota_profile is not None:
                self.append_parent(iota_profile)
                self.need_to_run_code = True

    @property
    def oita_profile(self):
        """
        Getter for the outer rotational transform profile (oita)

        Returns:
            ProfileSpec instance representing the oita profile
        """
        return self._oita_profile

    @oita_profile.setter
    def oita_profile(self, oita_profile):
        """
        Setter for the outer rotational transform profile (oita)

        Args:
            ProfileSpec instance for the outer rotational transform profile
        """

        if not isinstance(oita_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if oita_profile.dofs.full_x.size != self.nvol+1:
            raise ValueError('Invalid number of dofs. Should be equal to nvol+1!')

        if oita_profile is not self._oita_profile:
            logging.debug('Replacing oita_profile in setter')
            if self._oita_profile is not None:
                self.remove_parent(self._oita_profile)
            self._oita_profile = oita_profile
            if oita_profile is not None:
                self.append_parent(oita_profile)
                self.need_to_run_code = True

    @property
    def mu_profile(self):
        """
        Getter for the mu-profile

        Returns:
            ProfileSpec instance representing the mu profile
        """
        return self._mu_profile

    @mu_profile.setter
    def mu_profile(self, mu_profile):
        """
        Setter for the mu profile (oita)

        Args:
            ProfileSpec instance for the outer rotational transform profile
        """

        if not isinstance(mu_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if mu_profile.dofs.full_x.size != self.mvol:
            raise ValueError('Invalid number of dofs. Should be equal to Mvol!')

        if mu_profile is not self._mu_profile:
            logging.debug('Replacing mu_profile in setter')
            if self._mu_profile is not None:
                self.remove_parent(self._mu_profile)
            self._mu_profile = mu_profile
            if mu_profile is not None:
                self.append_parent(mu_profile)
                self.need_to_run_code = True

    @property
    def pflux_profile(self):
        """
        Getter for the poloidal flux profile (pflux)

        Returns:
            ProfileSpec instance representing the poloidal flux profile
        """
        return self._pflux_profile

    @pflux_profile.setter
    def pflux_profile(self, pflux_profile):
        """
        Setter for the poloidal flux profile (pflux)

        Args:
            ProfileSpec instance for the poloidal flux profile
        """

        if not isinstance(pflux_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if pflux_profile.dofs.full_x.size != self.mvol:
            raise ValueError('Invalid number of dofs. Should be equal to Mvol!')

        # pflux is a cumulative property
        pflux_profile.cumulative = True

        if pflux_profile is not self._pflux_profile:
            logging.debug('Replacing pflux_profile in setter')
            if self._pflux_profile is not None:
                self.remove_parent(self._pflux_profile)
            self._pflux_profile = pflux_profile
            if pflux_profile is not None:
                self.append_parent(pflux_profile)
                self.need_to_run_code = True

    @property
    def tflux_profile(self):
        """
        Getter for the toroidal flux profile (tflux)

        Returns:
            ProfileSpec instance representing the toroidal flux profile
        """
        return self._tflux_profile

    @tflux_profile.setter
    def tflux_profile(self, tflux_profile):
        """
        Setter for the toroidal flux profile (tflux)

        Args:
            ProfileSpec instance for the toroidal flux profile
        """
        if not isinstance(tflux_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if tflux_profile.dofs.full_x.size != self.mvol:
            raise ValueError('Invalid number of dofs. Should be equal to Mvol!')

        # pflux is a cumulative property
        tflux_profile.cumulative = True

        if tflux_profile is not self._tflux_profile:
            logging.debug('Replacing tflux_profile in setter')
            if self._tflux_profile is not None:
                self.remove_parent(self._tflux_profile)
            self._tflux_profile = tflux_profile
            if tflux_profile is not None:
                self.append_parent(tflux_profile)
                self.need_to_run_code = True

    @property
    def helicity_profile(self):
        """
        Getter for the magnetic helicity profile (helicity)

        Returns:
            ProfileSpec instance representing the magnetic helicity profile
        """
        return self._helicity_profile

    @helicity_profile.setter
    def helicity_profile(self, helicity_profile):
        """
        Setter for the toroidal flux profile (tflux)

        Args:
            ProfileSpec instance for the toroidal flux profile
        """

        if not isinstance(helicity_profile, ProfileSpec):
            raise ValueError('Input should be a ProfileSpec')

        # Check size
        if helicity_profile.dofs.full_x.size != self.mvol:
            raise ValueError('Invalid number of dofs. Should be equal to Mvol!')

        if helicity_profile is not self._helicity_profile:
            logging.debug('Replacing helicity_profile in setter')
            if self._helicity_profile is not None:
                self.remove_parent(self._tflux_profile)
            self._helicity_profile = helicity_profile
            if helicity_profile is not None:
                self.append_parent(helicity_profile)
                self.need_to_run_code = True

    def activate_profile(self, longname):
        """
        Take a profile from the inputlist, and make it optimizable in 
        simsopt.
        Args:
            longname: string, either 
                - 'pressure'
                - 'volume_current'
                - 'interface_current'
                - 'iota'
                - 'oita'
                - 'mu'
                - 'pflux'
                - 'tflux'
                - 'helicity'
        """
        profile_dict = {
            'pressure':        {'specname': 'pressure', 'cumulative': False, 'length': self.nvol},
            'volume_current':  {'specname': 'ivolume', 'cumulative': True, 'length': self.nvol},
            'interface_current': {'specname': 'isurf', 'cumulative': False, 'length': self.nvol-1},
            'helicity':        {'specname': 'helicity', 'cumulative': False, 'length': self.mvol},
            'iota':            {'specname': 'iota', 'cumulative': False, 'length': self.mvol},
            'oita':            {'specname': 'oita', 'cumulative': False, 'length': self.mvol},
            'mu':              {'specname': 'mu', 'cumulative': False, 'length':   self.mvol},
            'pflux':           {'specname': 'pflux', 'cumulative': True, 'length': self.mvol},
            'tflux':           {'specname': 'tflux', 'cumulative': True, 'length': self.mvol}
        }

        profile_data = self.inputlist.__getattribute__(profile_dict[longname]['specname'])[0:profile_dict[longname]['length']]
        profile = ProfileSpec(profile_data, cumulative=profile_dict[longname]['cumulative'], psi_edge=self.inputlist.phiedge)
        profile.unfix_all()
        self.__setattr__(longname + '_profile', profile)

    def set_profile(self, longname, lvol, value):
        """
        This function is used to set the pressure, currents, iota, oita,
        mu pflux and/or tflux in volume lvol

        Args:
            longname: string, either 
                - 'pressure'
                - 'volume_current'
                - 'surface_current'
                - 'iota'
                - 'oita'
                - 'mu'
                - 'pflux'
                - 'tflux'
                - 'helicity'
            lvol: integer, from 0 to Mvol-1
            value: real, new value
        """
        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        # If the profile is cumulative, values in lvol to Mvol-1 are modified.
        # If it is not cumulative, only the value in lvol is modified.
        if profile.cumulative: 
            old_value = profile.f(lvol)

            profile.set(lvol, value)
            for ivol in range(lvol + 1, self.mvol):
                profile.set(ivol, profile.f(ivol) - old_value + value)
        else:
            profile.set(lvol, value)

    def get_profile(self, longname, lvol):
        """
        This function is used to get the pressure, currents, iota, oita,
        mu pflux and/or tflux profiles.

        Args:
            longname: string, either
                - 'pressure'
                - 'volume_current'
                - 'surface_current'
                - 'iota'
                - 'oita'
                - 'mu'
                - 'pflux'
                - 'tflux'
                - 'helicity'
            lvol: integer, list or np.array of volume indices, from 0 to Mvol-1
        Returns:
            np.array of length lvol, with the profiles values.
        """

        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        return profile.f(lvol)

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def get_dofs(self):
        return np.array([self.inputlist.phiedge,
                         self.inputlist.curtor])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.inputlist.phiedge = x[0]
        self.inputlist.curtor = x[1]

        profiles = [
            self._volume_current_profile,
            self._interface_current_profile,
            self._pressure_profile,
            self._iota_profile,
            self._oita_profile,
            self._mu_profile,
            self._pflux_profile,
            self._tflux_profile,
            self._helicity_profile
        ]
        for p in profiles:
            if p is not None:
                p.psi_edge = x[0]

    @property 
    def poloidal_current_amperes(self):
        """
        return the total poloidal current in Amperes,
        i.e. the current that must be carried by the coils that link the plasma 
        poloidally
        """
        return self.inputlist.curpol/(mu_0)
    
    def _clear_f90wrap_array_caches(self):
        """
        Clear the f90wrap array caches. This is necessary when a new file is read after SPEC has run before.
        
        See https://github.com/hiddenSymmetries/simsopt/pull/431

        This function is for addressing the issue https://github.com/hiddenSymmetries/simsopt/issues/357
        """
        self.allglobal._arrays = {}
        self.inputlist._arrays = {}

    def _init_fortran_state(self, filename: str):
        """
        Initialize SPEC fortran state from an input file.

        Args:
            filename: Name of the file to load. It should end in ``.sp``.
        """
        logger.debug("Entering init")
        if self.mpi.proc0_groups:
            self.inputlist.initialize_inputs()
            logger.debug("Done with initialize_inputs")
            self.extension = filename[:-3]  # Remove the ".sp"
            self.allglobal.ext = self.extension
            self.allglobal.read_inputlists_from_file()
            logger.debug("Done with read_inputlists_from_file")
            self.allglobal.check_inputs()

        logger.debug('About to call broadcast_inputs')
        self.allglobal.broadcast_inputs()
        logger.debug('About to call preset')
        self.lib.preset()
        logger.debug("Done with init")

    def run(self, update_guess: bool = True):
        """
        Run SPEC, if needed.
        The FORTRAN state has been created in the __init__ method (there can be only one for one python kernel), and the values in the FORTRAN memory are updated by python. 

        The most important values are in inputlist and allglobal. 

        Note: Since all calls to fortran memory access the same memory, there is no difference between spec.inputlist and self.inputlist (or even my_other_spec2.inputlist).

        Args:
            - update_guess: boolean. If True, initial guess will be updated with
              the geometry of the interfaces found at equilibrium. Default is 
              True
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run SPEC.")
            return
        if self.mpi.proc0_groups:
            logger.info(f"Group {self.mpi.group}: Preparing to run SPEC.")
        self.counter += 1

        si = self.inputlist  # Shorthand

        # if freeboundary, store plasma-caused boundary field to re-set if run does not converge
        if self.freebound:
            initial_bns = np.copy(si.bns)
            initial_bnc = np.copy(si.bnc)

        # Check that number of volumes in internal memory is consistent with
        # the input file
        if self.nvol != si.nvol:
            raise ValueError('Inconsistent Nvol')

        # nfp must be consistent between the surface and SPEC. The surface's
        # value trumps.
        si.nfp = self.boundary.nfp
        si.istellsym = int(self.boundary.stellsym)

        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()

        # Transfer boundary data to fortran:
        si.rbc[:, :] = self.array_translator(boundary_RZFourier.rc, style='simsopt').as_spec
        si.zbs[:, :] = self.array_translator(boundary_RZFourier.zs, style='simsopt').as_spec
        si.rbs[:, :] = 0.0
        si.zbc[:, :] = 0.0
        if not self.stellsym:
            si.rbs[:, :] = self.array_translator(boundary_RZFourier.rs, style='simsopt').as_spec
            si.zbc[:, :] = self.array_translator(boundary_RZFourier.zc, style='simsopt').as_spec

        # transfer normal field to fortran:
        if self.freebound:
            si.vns[:, :] = self.array_translator(self.normal_field.get_vns_asarray(), style='simsopt').as_spec
            if not self.stellsym:
                si.vnc[:, :] = self.array_translator(self.normal_field.get_vnc_asarray(), style='simsopt').as_spec

        # Set the coordinate axis using the lrzaxis=2 feature:
        si.lrzaxis = 2

        # Set axis from latest converged state
        mn = self.axis['rac'].size
        si.rac[0:mn] = self.axis['rac']
        si.zas[0:mn] = self.axis['zas']
        if si.istellsym == 0:
            si.ras[0:mn] = self.axis['ras']
            si.zac[0:mn] = self.axis['zac']

        # Set initial guess
        if self.initial_guess is not None:  # note: self.initial_guess is None for workers!! only leaders read and do_stuff with initial_guess
            self._set_spec_initial_guess()  # workers get the info through a broadcast. this line fails if workers get a guess set

            # write the boundary which is a guess in freeboundary
            if self.freebound:
                boundaryguess = self.initial_guess[-1].to_RZFourier()
                si.rbc[:] = self.array_translator(boundaryguess.rc, style='simsopt').as_spec
                si.zbs[:] = self.array_translator(boundaryguess.zs, style='simsopt').as_spec
                if not self.stellsym:
                    si.rbs[:] = self.array_translator(boundaryguess.rs, style='simsopt').as_spec
                    si.zbc[:] = self.array_translator(boundaryguess.zc, style='simsopt').as_spec

        # Set profiles from dofs
        if self.pressure_profile is not None:
            si.pressure[0:self.nvol] = self.pressure_profile.get(
                np.arange(0, self.nvol))
            if si.lfreebound:
                si.pressure[self.nvol] = 0

        if self.volume_current_profile is not None:
            # Volume current is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_ivolume = copy.copy(si.ivolume)
            for lvol in range(0, self.mvol):
                if self.volume_current_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.ivolume[lvol] = si.ivolume[lvol] - \
                            old_ivolume[lvol - 1] + si.ivolume[lvol - 1]
                        self.set_profile(
                            'volume_current', lvol=lvol, value=si.ivolume[lvol])
                else:
                    si.ivolume[lvol] = self.get_profile('volume_current', lvol)

            if si.lfreebound:
                si.ivolume[self.nvol] = si.ivolume[self.nvol - 1]
                self.volume_current_profile.set(
                    key=self.mvol - 1, new_val=si.ivolume[self.nvol - 1])

        if self.interface_current_profile is not None:
            si.isurf[0:self.mvol - 1] = \
                self.interface_current_profile.get(np.arange(0, self.mvol-1))

        # Update total plasma toroidal current in case of freeboundary
        # calculation
        if ((self.volume_current_profile is not None) or
            (self.interface_current_profile is not None)) and \
                si.lfreebound:
            si.curtor = si.ivolume[self.nvol - 1] + np.sum(si.isurf)

        if self.iota_profile is not None:
            si.iota[0:self.nvol+1] = self.iota_profile.get(np.arange(0, self.nvol+1))

        if self.oita_profile is not None:
            si.oita[0:self.nvol+1] = self.oita_profile.get(np.arange(0, self.nvol+1))

        if self.mu_profile is not None:
            si.mu[0:self.nvol] = self.mu_profile.get(np.arange(0, self.nvol))
            if si.lfreebound:
                si.mu[self.mvol] = 0

        if self.pflux_profile is not None:
            # Pflux is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_pflux = copy.copy(si.pflux)
            for lvol in range(0, self.mvol):
                if self.pflux_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.pflux[lvol] = si.pflux[lvol] - \
                            old_pflux[lvol - 1] + si.pflux[lvol - 1]
                        self.pflux_profile.set(
                            key=lvol, new_val=si.pflux[lvol])
                else:
                    si.pflux[lvol] = self.pflux_profile.get(lvol)

        if self.tflux_profile is not None:
            # tflux is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_tflux = copy.copy(si.tflux)
            for lvol in range(0, self.mvol):
                if self.tflux_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.tflux[lvol] = si.tflux[lvol] - \
                            old_tflux[lvol - 1] + si.tflux[lvol - 1]
                        self.tflux_profile.set(
                            key=lvol, new_val=si.tflux[lvol])
                else:
                    si.tflux[lvol] = self.tflux_profile.get(lvol)

        if self.helicity_profile is not None:
            si.helicity[0:self.nvol] = self.helicity_profile.get(np.arange(0, self.nvol))
            if si.lfreebound:
                si.helicity[self.mvol] = 0

        # Another possible way to initialize the coordinate axis: use
        # the m=0 modes of the boundary.
        # m = 0
        # for n in range(2):
        #     si.rac[n] = si.rbc[n + si.mntor, m + si.mmpol]
        #     si.zas[n] = si.zbs[n + si.mntor, m + si.mmpol]
        filename = self.extension + \
            '_{:03}_{:06}'.format(self.mpi.group, self.counter)
        logger.info("Running SPEC using filename " + filename)
        self.allglobal.ext = filename
        try:
            # Here is where we actually run SPEC:
            if self.mpi.proc0_groups:
                logger.debug('About to call check_inputs')
                self.allglobal.check_inputs()
            logger.debug('About to call broadcast_inputs')
            self.mpi.comm_groups.Barrier()  # Barrier to ensure all groups are ready to broadcast. 
            self.allglobal.broadcast_inputs()
            logger.debug('About to call preset')
            self.lib.preset()
            logger.debug('About to call init_outfile')
            self.lib.sphdf5.init_outfile()
            logger.debug('About to call mirror_input_to_outfile')
            self.lib.sphdf5.mirror_input_to_outfile()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                self.allglobal.wrtend()
            logger.debug('About to call init_convergence_output')
            self.lib.sphdf5.init_convergence_output()
            logger.debug('About to call spec')
            self.lib.spec()
            logger.debug('About to call diagnostics')
            self.lib.final_diagnostics()
            logger.debug('About to call write_grid')
            self.lib.sphdf5.write_grid()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                self.allglobal.wrtend()
            logger.debug('About to call hdfint')
            self.lib.sphdf5.hdfint()
            logger.debug('About to call finish_outfile')
            self.lib.sphdf5.finish_outfile()
            logger.debug('About to call ending')
            self.lib.ending()

        except BaseException:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure("SPEC did not run successfully.")
        if self.mpi.proc0_groups:
            logger.info(f"{filename}: SPEC run complete.")
        # Barrier so workers do not try to read the .h5 file before it
        # is finished:
        self.mpi.comm_groups.Barrier()

        # Try to read SPEC output. 
        try:
            self.results = py_spec.SPECout(filename + '.sp.h5')
        except BaseException:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure(
                "Unable to read results following SPEC execution")

        if self.mpi.proc0_groups:
            logger.info("Successfully loaded SPEC results.")
        self.need_to_run_code = False

        # Deal with unconverged equilibria - these are excluded by 
        # the optimizer, and the objective function is set to a large number
        if self.results.output.ForceErr > self.tolerance:
            # re-set the boundary field guess before failing, so the next run does not start unconverged. 
            if self.freebound:
                logger.info(f"run {filename}: Force balance unconverged, re-setting boundary field guess and raising ObjectiveFailure")
                si.bns, si.bnc = initial_bns, initial_bnc
            raise ObjectiveFailure(
                'SPEC could not find force balance'
            )
        
        # check for picard convergence once SPEC is updated
        if self.freebound:
            try: 
                if self.results.output.BnsErr > self.inputlist.gbntol:
                    if self.mpi.proc0_groups:
                        logger.info(f"run {filename}: Picard iteration unconverged, re-setting boundary field guess and raising ObjectiveFailure")
                    si.bns, si.bnc = initial_bns, initial_bnc
                    raise ObjectiveFailure(
                        'Picard iteration failed to reach convergence, increase mfreeits or gbntol'
                    )
            except AttributeError:
                logger.info('Picard convergence not checked, please update SPEC version 3.22 or higher')
                print("Picard convergence not checked, please update SPEC version 3.22 or higher")
            
        # Save geometry as initial guess for next iterations
        if update_guess:
            new_guess = None
            if self.mvol > 1:
                new_guess = [
                    SurfaceRZFourier(nfp=si.nfp, stellsym=si.istellsym, mpol=si.mpol, ntor=si.ntor) for n in range(0, self.mvol-1)
                ]

                for ii, (mm, nn) in enumerate(zip(self.results.output.im, self.results.output.in_)):
                    nnorm = (nn / si.nfp).astype('int')  # results.output.in_ is fourier number for 1 field period for reasons...
                    for lvol in range(0, self.mvol-1):
                        new_guess[lvol].set_rc(mm, nnorm, self.results.output.Rbc[lvol+1, ii])
                        new_guess[lvol].set_zs(mm, nnorm, self.results.output.Zbs[lvol+1, ii])

                        if not si.istellsym:
                            new_guess[lvol].set_rs(mm, nnorm, self.results.output.Rbs[lvol+1, ii])
                            new_guess[lvol].set_zc(mm, nnorm, self.results.output.Zbc[lvol+1, ii])

                axis = {}
                axis['rac'] = self.results.output.Rbc[0, 0:si.ntor+1]
                axis['zas'] = self.results.output.Zbs[0, 0:si.ntor+1]
                self.axis = copy.copy(axis)

            # Enforce SPEC to use initial guess, but only group leaders have the necessary arrays
            if self.mpi.proc0_groups:
                self.initial_guess = new_guess  # set this here? or make whole above monster proc0-exclusive
            self.inputlist.linitialize = 0

        # Group leaders handle deletion of files:
        if self.mpi.proc0_groups:
            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.mpi.group > 0):
                os.remove(filename + '.sp.h5')
                os.remove(filename + '.sp.end')
                if os.path.exists('.' + filename + '.sp.DF'):
                    os.remove('.' + filename + '.sp.DF')

            # Delete the previous output file, if desired:
            for file_to_delete in self.files_to_delete:
                os.remove(file_to_delete)
            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.mpi.group == 0) and (
                    self.counter > 0) and (not self.keep_all_files):
                self.files_to_delete.append(filename + '.sp.h5')
                self.files_to_delete.append(filename + '.sp.end')
        if self.mpi.proc0_groups:
            logger.info(f"file {filename} completed run sucessfully")

    def volume(self):
        """
        Return the volume inside the boundary flux surface.
        """
        self.run()
        return self.results.output.volume * self.results.input.physics.Nfp

    def iota(self):
        """
        Return the rotational transform in the middle of the volume.
        """
        self.run()
        return self.results.transform.fiota[1, 0]
    

    def vacuum_well(self):
        """
        Compute a single number W that summarizes the vacuum magnetic well,
        given by the formula

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the radial coordinate s. Positive values of W are
        favorable for stability to interchange modes. This formula for
        W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius. To compute dV/ds, we use the py_spec package to get 
        d/ds sqrtg, compute dV/ds by integrating over the flux surface and
        take the finite differences dV/ds(s=0) - dV/ds(s=1).
        
        """
        self.run()


        
        def get_sqrtg_fourier(lvol=0, sarr=np.linspace(1, 1, 1), tarr=np.linspace(0, 0, 1), zarr=np.linspace(0, 0, 1), input1D=False, derivative=False):
            nax = np.newaxis
            sym = self.inputlist.istellsym == 1

            Rac, Rbc = self.results.output.Rbc[lvol : lvol + 2]
            Ras, Rbs = self.results.output.Rbs[lvol : lvol + 2]
            Zas, Zbs = self.results.output.Zbs[lvol : lvol + 2]
            Zac, Zbc = self.results.output.Zbc[lvol : lvol + 2]

            mn = Rac.size  # s.output.mn
            im = self.results.output.im
            in_ = self.results.output.in_    
            #sbar = (sarr + 1) / 2
            sbar = np.divide(np.add(sarr, 1.0), 2.0)
            fac = []

            # We are only iterested in m=0, n=0
            sin = 0
            cos = 1

            Igeometry = self.inputlist.igeometry
            rpol = self.inputlist.rpol
            rtor = self.inputlist.rtor

            assert im.shape[0] == mn
            assert len(im.shape) == 1
            assert in_.shape[0] == mn
            assert Rbc.shape[0] == mn


            if Igeometry == 1:
                for j in range(mn):
                    fac.append([sbar, 0.5 * np.ones(sarr.size), np.zeros(sarr.size)])
            elif Igeometry == 2:
                for j in range(mn):
                    if lvol > 0 or im[j] == 0:
                        fac.append([sbar, 0.5 * np.ones(sarr.size), np.zeros(sarr.size)])
                    else:
                        fac.append(
                            [
                                sbar ** (im[j] + 1.0),
                                (im[j] + 1.0) / 2.0 * sbar ** (im[j]),
                                (im[j] + 1.0) * (im[j]) / 4.0 * sbar ** (im[j] - 1),
                            ]
                        )
            elif Igeometry == 3:
                for j in range(mn):
                    if lvol == 0 and im[j] == 0:
                        fac.append([sbar ** 2, sbar, 0.5 * np.ones(sarr.size)])
                    elif lvol == 0 and im[j] > 0:
                        fac.append(
                            [
                                sbar ** im[j],
                                (im[j] / 2.0) * sbar ** (im[j] - 1.0),
                                (im[j] * (im[j] - 1) / 4.0) * sbar ** (im[j] - 2.0),
                            ]
                        )
                    else:
                        fac.append([sbar, 0.5 * np.ones(sarr.size), np.zeros(sarr.size)])
            # now fac has the dimension (number of modes, number of derivatives, number of s points)
            fac = np.array(fac)
            # transpose to (number of derivatives, number of modes, number of s points)
            fac = np.moveaxis(np.array(fac), 0, 1)

            if not input1D:
                im = im[:, nax]
                in_ = in_[:, nax]

                Rac = Rac[:, nax]
                Rbc = Rbc[:, nax]
                Zas = Zas[:, nax]
                Zbs = Zbs[:, nax]
                if not sym:
                    Ras = Ras[:, nax]
                    Rbs = Rbs[:, nax]
                    Zac = Zac[:, nax]
                    Zbc = Zbc[:, nax]

            dR1 = Rac + fac[0] * (Rbc - Rac)
            Rarr0 = dR1 * cos

            Rarr1 = fac[1] * (Rbc - Rac) * cos
            Rarr2 = -im * dR1 * sin
            
            # We only need Z for Igeometry=3
            if Igeometry == 3:
                dZ1 = Zas + fac[0] * (Zbs - Zas)
                Zarr1 = fac[1] * (Zbs - Zas) * sin
                Zarr2 = im * dZ1 * cos
            else:
                Zarr0 = None

            if Igeometry == 1:
                jacobian = Rarr1 * rpol * rtor
            elif Igeometry == 2:
                jacobian = Rarr1 * Rarr0
            elif Igeometry == 3:
                jacobian = Rarr0 * (Rarr2 * Zarr1 - Rarr1 * Zarr2)  # from matlab
            
            # for the inner and outer edge of this lvol
            return jacobian 

        # TODO: get these from the inputlist
        nt = 64
        nz = 64

        tarr = np.linspace(0, 2*np.pi, nt, endpoint=True)
        zarr = np.linspace(0, 2*np.pi / self.nfp, nz, endpoint=True)

        flux = np.atleast_1d(spec.inputlist.tflux[: self.nvol])
        # In Fortran tflux is normalized so that tflux(Nvol) = 1.0, so tflux[-1]
        # Get the flux per volume and weigh the contributions accordingly
        flux_per_volume = np.diff(flux, prepend=0) / flux[-1]

        # Integrate
        dt = tarr[1]-tarr[0]
        dz = zarr[1]-zarr[0]
        ds = 1e-3

        def dVds(ivol:int, s:float):
            j = self.results.jacobian(lvol=ivol, sarr=np.array([s-ds, s+ds]), tarr=tarr, zarr=zarr)
            volume = self.nfp * np.sum(j)*dt*dz*ds
            return volume / (2*ds)
 
        # Remapping from [-1,1] to [flux(s=-1), flux(s=1)]
        dVds_s0 = dVds(ivol=self.nvol-1, s=-1+ds) / flux_per_volume[self.nvol-1]*2
        dVds_s1 = dVds(ivol=self.nvol-1, s=1-ds) / flux_per_volume[self.nvol-1]*2

        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well

        # Compute vacuum well for each volume
        well = np.zeros(self.nvol)
        for ivol in range(self.nvol):
            # derivatives of sqrtg on the inner and outer edge. 
            # TODO: Ideally, we only need <d sqrtg/ds> averaged over the flux surface, i.e. sqrtg[0,0] in fourier representation.  
            _, _, jac, g, djacobian, _ = self.results.get_grid_and_jacobian_and_metric(lvol=ivol, sarr=np.array([-0.99, 1]), tarr=tarr, zarr=zarr, derivative=True)
            
            def surface_integral(array:np.ndarray):
                return np.sum(array)*dt*dz
            
            dVds_s0 =  surface_integral(djacobian[0,...,0])
            dVds_s1 =  surface_integral(djacobian[1,...,0])
            well[ivol] = (dVds_s0 - dVds_s1) / dVds_s0

            gmnc00 = get_sqrtg_fourier(lvol=ivol, sarr=np.array([-0.99, 1]), tarr=tarr, zarr=zarr, derivative=True)
            dVds = 4 * np.pi * np.pi * np.abs(gmnc00[0, 1:])

            # assert dVds_s0 == dVds[0]
            # assert dVds_s1 == dVds[-1]
            dVds_s0 = dVds[0]
            dVds_s1 = dVds[-1]


        
        weighted_well =  well # * flux_per_volume
        return weighted_well
    
    def array_translator(self, array=None, style='spec'):
        """
        Returns a SpecFourierArray object to help transforming between
        arrays using SPEC conventions and simsopt conventions.

        create from an array of either style by setting style='spec' or
        style='simsopt' upon creation. 
        
        The created class will have properties:
        - as_spec: returns the SPEC style array
        - as_simsopt: returns the simsopt style array
        and setters to set the arrays from either convention after creation.

        SPEC arrays are zero-padded and indexed with Fortran conventions, 
        in python that means the m=0,n=0 component is at [nmtor,mmpol].

        use: 
        vnc = myspec.array_translator(myspec.inputlist.rbc)
        mysurf.rc = vnc.as_simsopt 
        """
        if style == 'spec':
            return Spec.SpecFourierArray(self, array)
        elif style == 'simsopt':
            obj = Spec.SpecFourierArray(self)
            obj.as_simsopt = array  # setter function
            return obj
        else: 
            raise ValueError(f'array style:{style} not supported, options: [simsopt, spec]')
            
    # Inner class of Spec
    class SpecFourierArray: 
        """
        Helper class to make index-jangling easier and hide away
        strange SPEC array sizing and indexing conventions.

        SPEC stores Fourier series in a very large zero-padded
        array with Fortran variable indexing and 'n,m' indexing 
        convention.
        In python this means the m=0,n=0 component is at [nmtor,mmpol]. 


        """
        def __init__(self, outer_spec, array=None): 
            self._outer_spec = outer_spec
            self._mmpol = outer_spec.inputlist.mmpol
            self._mntor = outer_spec.inputlist.mntor
            if array is None:
                array = np.zeros((2*self._mntor+1, 2*self._mmpol+1))   
            self._array = np.array(array)
        
        @property
        def as_spec(self):
            return self._array
        
        @as_spec.setter
        def as_spec(self, array):
            if array.shape != (2*self._mntor+1, 2*self._mmpol+1):
                raise ValueError('Array size is not consistent witn mmpol and mntor')
            self._array = array

        @property
        def as_simsopt(self, ntor=None, mpol=None):
            """
            get a simsopt style 'n,m' intexed non-padded array 
            from the SpecFourierArray.
            """
            if ntor is None:
                ntor = self._outer_spec.ntor
            if mpol is None:
                mpol = self._outer_spec.mpol
            n_start = self._mntor - ntor
            n_end = self._mntor + ntor + 1
            m_start = self._mmpol
            m_end = self._mmpol + mpol + 1
            return self._array[n_start:n_end, m_start:m_end].transpose()  # switch n and m
        
        @as_simsopt.setter
        def as_simsopt(self, array, ntor=None, mpol=None):
            """
            set the SpecFourierArray from a simsopt style 'm,n' intexed non-padded array
            """
            if mpol is None:
                mpol = array.shape[0]-1
            if ntor is None:
                ntor = int((array.shape[1]-1)/2)
            n_start = self._mntor - ntor
            n_end = self._mntor + ntor + 1
            m_start = self._mmpol
            m_end = self._mmpol + mpol + 1
            self._array = np.zeros((2*self._mntor+1, 2*self._mmpol+1))
            self._array[n_start:n_end, m_start:m_end] = array.transpose()


class Residue(Optimizable):
    """
    Greene's residue, evaluated from a Spec equilibrum

    Args:
        spec: a Spec object
        pp, qq: Numerator and denominator for the resonant iota = pp / qq
        vol: Index of the Spec volume to consider
        theta: Spec's theta coordinate at the periodic field line
        s_guess: Guess for the value of Spec's s coordinate at the periodic
                field line
        s_min, s_max: bounds on s for the search
        rtol: the relative tolerance of the integrator
    """

    def __init__(self, spec, pp, qq, vol=1, theta=0., s_guess=None, s_min=-1.0,
                 s_max=1.0, rtol=1e-9):
        # if not spec_found:
        if spec is None:
            raise RuntimeError(
                "Residue requires spec package to be installed.")
        # if not pyoculus_found:
        if pyoculus is None:
            raise RuntimeError(
                "Residue requires pyoculus package to be installed.")

        self.spec = spec
        self.pp = pp
        self.qq = qq
        self.vol = vol
        self.theta = theta
        self.rtol = rtol
        if s_guess is None:
            self.s_guess = 0.0
        else:
            self.s_guess = s_guess
        self.s_min = s_min
        self.s_max = s_max
        self.depends_on = ['spec']
        self.need_to_run_code = True
        self.fixed_point = None
        # We may at some point want to allow Residue to use a
        # different MpiPartition than the Spec object it is attached
        # to, but for now we'll use the same MpiPartition for
        # simplicity.
        self.mpi = spec.mpi
        super().__init__(depends_on=[spec])

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def J(self):
        """
        Run Spec if needed, find the periodic field line, and return the residue
        """
        if self.need_to_run_code:
            self.spec.run()
        
        if not self.mpi.proc0_groups:
            logger.debug(
                "This proc is skipping Residue.J() since it is not a group leader.")
            return 0.0
        else:
            specb = pyoculus.problems.SPECBfield(self.spec.results, self.vol)
            # Set nrestart=0 because otherwise the random guesses in
            # pyoculus can cause examples/tests to be
            # non-reproducible.
            fp = pyoculus.solvers.FixedPoint(
                specb, {
                    'theta': self.theta, 'nrestart': 0}, integrator_params={
                    'rtol': self.rtol})
            self.fixed_point = fp.compute(self.s_guess,
                                          sbegin=self.s_min,
                                          send=self.s_max,
                                          pp=self.pp, qq=self.qq)
            self.need_to_run_code = False

        if self.fixed_point is None:
            raise ObjectiveFailure("Residue calculation failed")
        logger.info(f"group {self.mpi.group} found residue {self.fixed_point.GreenesResidue} for {self.pp}/{self.qq} in {self.spec.allglobal.ext}")

        return self.fixed_point.GreenesResidue


class QuasisymmetryRatioResidualSpec(Optimizable):
    r"""
    This class provides a measure of the deviation from quasisymmetry,
    one that can be computed without Boozer coordinates.  This metric
    is based on the fact that for quasisymmetry, the ratio

    .. math::
        (\vec{B}\times\nabla B \cdot\nabla\psi) / (\vec{B} \cdot\nabla B)

    is constant on flux surfaces.

    Specifically, this class represents the objective function

    .. math::
        f = \sum_{s_j} w_j \left< \left[ \frac{1}{B^3} \left( (N - \iota M)\vec{B}\times\nabla B\cdot\nabla\psi - (MG+NI)\vec{B}\cdot\nabla B \right) \right]^2 \right>

    where the sum is over a set of flux surfaces with normalized
    toroidal flux :math:`s_j`, the coefficients :math:`w_j` are
    user-supplied weights, :math:`\left< \ldots \right>` denotes a
    flux surface average, :math:`G(s)` is :math:`\mu_0/(2\pi)` times
    the poloidal current outside the surface, :math:`I(s)` is
    :math:`\mu_0/(2\pi)` times the toroidal current inside the
    surface, :math:`\mu_0` is the permeability of free space,
    :math:`2\pi\psi` is the toroidal flux, and :math:`(M,N)` are
    user-supplied integers that specify the desired helicity of
    symmetry. If the magnetic field is quasisymmetric, so
    :math:`B=B(\psi,\chi)` where :math:`\chi=M\vartheta - N\varphi`
    where :math:`(\vartheta,\varphi)` are the poloidal and toroidal
    Boozer angles, then :math:`\vec{B}\times\nabla B\cdot\nabla\psi
    \to -(MG+NI)(\vec{B}\cdot\nabla\varphi)\partial B/\partial \chi`
    and :math:`\vec{B}\cdot\nabla B \to (-N+\iota
    M)(\vec{B}\cdot\nabla\varphi)\partial B/\partial \chi`, implying
    the metric :math:`f` vanishes. The flux surface average is
    discretized using a uniform grid in the SPEC poloidal and toroidal
    angles :math:`(\theta,\phi)`. In this case :math:`f` can be
    written as a finite sum of squares:

    .. math::
        f = \sum_{s_j, \theta_j, \phi_j} R(s_j, \theta_k, \phi_{\ell})^2

    where the :math:`\phi_{\ell}` grid covers a single field period.
    Here, each residual term is

    .. math::
        R(\theta, \phi) = \sqrt{w_j \frac{n_{fp} \Delta_{\theta} \Delta_{\phi}}{V'}|\sqrt{g}|}
        \frac{1}{B^3} \left( (N-\iota M)\vec{B}\times\nabla B\cdot\nabla\psi - (MG+NI)\vec{B}\cdot\nabla B \right).

    Here, :math:`n_{fp}` is the number of field periods,
    :math:`\Delta_{\theta}` and :math:`\Delta_{\phi}` are the spacing
    of grid points in the poloidal and toroidal angles,
    :math:`\sqrt{g} = 1/(\nabla s\cdot\nabla\theta \times
    \nabla\phi)` is the Jacobian of the :math:`(s,\theta,\phi)`
    coordinates, and :math:`V' = \int_0^{2\pi} d\theta \int_0^{2\pi}d\phi |\sqrt{g}| = dV/d\psi`
    where :math:`V` is the volume enclosed by a flux surface.

    Args:
        spec: A :obj:`simsopt.mhd.Spec` object from which the
          quasisymmetry error will be calculated.
        surfaces: Value of normalized toroidal flux at which you want the
          quasisymmetry error evaluated, or a list of values. Each
          value must be in the interval [0, 1], with 0 corresponding
          to the magnetic axis and 1 to the SPEC plasma boundary. If None,
          it will use the array of tflux values in the Spec inputlist as surfaces.
          This parameter corresponds to :math:`s_j` above.
        helicity_m: Desired poloidal mode number :math:`M` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
        helicity_n: Desired toroidal mode number :math:`\hat{N} = N / n_{fp}` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
          Note that the supplied value of ``helicity_n`` will be multiplied by
          the number of field periods :math:`n_{fp}`, so typically
          ``helicity_n`` should be +1 or -1 for quasi-helical symmetry.
        weights: The list of weights :math:`w_j` for each flux surface.
          If ``None``, a weight of :math:`w_j=1` will be used for
          all surfaces.
        ntheta: Number of grid points in :math:`\theta` used to
          discretize the flux surface average.
        nphi: Number of grid points per field period in :math:`\phi` used to
          discretize the flux surface average.
    """

    def __init__(self,
        spec: Spec,
        surfaces: np.ndarray = None,
        helicity_m: int = 1,
        helicity_n: int = 0,
        weights: np.ndarray = None,
        ntheta: int = 63,
                 nphi: int = 64) -> None:

        self.spec = spec

        self.ntheta = ntheta
        self.nphi = nphi
        self.helicity_m = helicity_m
        self.helicity_n = helicity_n


        if surfaces is None:
            flux = np.atleast_1d(spec.inputlist.tflux[: spec.nvol])
            # In Fortran tflux is normalized so that tflux(Nvol) = 1.0, so tflux[-1]
            normalized_toroidal_flux = flux / flux[-1]
            self.surfaces = normalized_toroidal_flux
            self.iota_on_volume_interfaces = True
        else:
            self.surfaces = np.atleast_1d(surfaces)
            self.iota_on_volume_interfaces = False
        if (self.surfaces < 0).any() or (self.surfaces > 1).any():
            raise ValueError("Surfaces for evaluating QuasisymmetryRatioResidual must be in the range s in [0, 1]")
        logger.debug("Surfaces: %s", self.surfaces)

        if weights is None:
            self.weights = np.ones(self.surfaces.shape)
        else:
            self.weights = weights
        assert len(self.weights) == len(self.surfaces)
        super().__init__(depends_on=[spec])

    # def recompute_bell(self, parent=None):
    #     self.need_to_run_code = True

    def compute(self):
        """
        Compute the quasisymmetry metric. This function returns an object
        that contains (as attributes) all the intermediate quantities
        for the calculation. Users do not need to call this function
        for optimization; instead the :func:`residuals()` function can be
        used. However, this function can be useful if users wish to
        inspect the quantities going into the calculation.
        """
        spec = self.spec
        spec.run()
        if not spec.stellsym:
            raise RuntimeError(
                "Quasisymmetry class cannot yet handle non-stellarator-symmetric configs"
            )
        
        logger.debug('Evaluating quasisymmetry residuals')
        ns = len(self.surfaces)
        ntheta = self.ntheta
        nphi = self.nphi
        nfp = spec.nfp
        d_psi_d_s = -self.spec.inputlist.phiedge / (2 * np.pi)
        import warnings
        # FIXME: only for nvol = 1
        d_psi_d_s = 2 * d_psi_d_s # 2 is the differential element for change of variables from s[-1, 1] to s[0, 1] d_psi_d_s 
        warnings.warn("d_psi_d_s is not correct for nvol > 1")

        flux = np.atleast_1d(spec.inputlist.tflux[: spec.nvol])
        # In Fortran tflux is normalized so that tflux(Nvol) = 1.0, so tflux[-1]
        normalized_toroidal_flux = flux / flux[-1]
        if self.iota_on_volume_interfaces:
            self.surfaces = normalized_toroidal_flux

        bsupu = []
        bsupv = []
        bsubu = []
        bsubv = []
        d_B_d_theta = []
        d_B_d_phi = []
        modB = []
        sqrtg = []

        # differential element we need due to the change of variables from volume local coordinates to flux coordinates
        ds = np.zeros(ns) 
        # Closed interval [0, 1] for s
        min_tflux = 0.0 - np.finfo(float).eps
        for lvol, max_tflux in zip(range(spec.nvol), normalized_toroidal_flux):
            # Find all surfaces that are in the current volume (<= max_tflux)
            mask = (self.surfaces <= max_tflux) & (self.surfaces > min_tflux )
            s_in_volume_bounds = self.surfaces[mask]
            # Due to rescaling of s=[-1,1] range of the chebyshev basis in SPEC to the flux coordinate range
            local_s = 2*(s_in_volume_bounds-min_tflux)/(max_tflux-min_tflux)-1
            ds[mask] = 2/(max_tflux-min_tflux)
            min_tflux = max_tflux
            if len(local_s) == 0:
                continue

            theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
            # jacobian = sqrtg
            Rarr0, Zarr0, local_sqrtg, g, djacobian, dg = spec.results.get_grid_and_jacobian_and_metric(lvol, sarr=local_s,tarr=theta1d, zarr=phi1d, derivative=True)
            computed_sqrtg = np.sqrt(np.linalg.det(g.reshape((-1,3,3))))
            assert np.allclose(local_sqrtg, computed_sqrtg.reshape(local_sqrtg.shape))


            j2 = spec.results.jacobian(lvol=lvol, sarr=local_s,tarr=theta1d, zarr=phi1d)
            
            # transform local s to global s
            print("s_in_volume_bounds",s_in_volume_bounds)
            # local_sqrtg = local_sqrtg / s_in_volume_bounds[:,None,None]

            # B^s, B^theta, B^zeta
            Bcontrav, dBcontrav = spec.results.get_B(lvol, local_sqrtg, sarr=local_s, tarr=theta1d, zarr=phi1d, derivative=True)
            Bcov = spec.results.get_B_covariant(Bcontrav, g)
            local_modB, dmodB2 = spec.results.get_modB(Bcontrav, g, derivative=True, dBcontrav=dBcontrav, dg=dg)
            # Chain rule: \nabla (|B^2|)^1/2 = 1/2 * |B^2|^-1/2 * \nabla |B^2| = 1/2 * |B|^-1 * \nabla |B^2|
            # and \nabla |B^2| = dmodB2
            dmodB = np.einsum("...i,...ij->...ij", 0.5 / local_modB, dmodB2)
            
            # This means that some surfaces are intersecting
            if np.any(local_sqrtg < 0):
                raise ObjectiveFailure("QuasisymmetryRatioResidualSpec failed, sqrtg < 0, Some surfaces are intersecting")

            # bsups = B^s, bsupu = B^phi, bsupv = B^zeta
            bsupu.append(Bcontrav[..., 1])
            bsupv.append(Bcontrav[..., 2])
            bsubu.append(Bcov[..., 1])
            bsubv.append(Bcov[..., 2])
            d_B_d_theta.append(dmodB[..., 1])
            d_B_d_phi.append(dmodB[..., 2])
            modB.append(local_modB)
            sqrtg.append(-1 * local_sqrtg)

        assert len(bsupu) > 0 

        bsupu = np.concatenate(bsupu, axis=0)
        bsupv = np.concatenate(bsupv, axis=0)
        bsubu = np.concatenate(bsubu, axis=0)
        bsubv = np.concatenate(bsubv, axis=0)
        d_B_d_theta = np.concatenate(d_B_d_theta, axis=0)
        d_B_d_phi = np.concatenate(d_B_d_phi, axis=0)
        modB = np.concatenate(modB, axis=0)
        sqrtg = np.concatenate(sqrtg, axis=0)

        # Either get the prescribed iota or the iota from poincae output, depending on the setting
        if self.iota_on_volume_interfaces or not hasattr(spec.results, "transform") or np.shape(spec.results.transform.fiota)[1] <= 1: 
            # Include iota on axis
            iota = spec.inputlist.iota[: spec.nvol + 1]
            interface_positions_s = np.pad(normalized_toroidal_flux, (1, 0))
            assert iota.shape == interface_positions_s.shape
            interp = interp1d(interface_positions_s, iota, fill_value="extrapolate")
            iota = interp(self.surfaces)
        else:
            s_transform = spec.results.transform.fiota[0, spec.results.poincare.success == 1]
            iota_transform = spec.results.transform.fiota[1, spec.results.poincare.success == 1]
            # If the field line integration fails, we don't have any valid iota values. prevent NaN
            if len(iota_transform) <= 1:
                raise ObjectiveFailure("QuasisymmetryRatioResidualSpec failed, could not find iota values")
            else:
                # First, interpolate in s to get the quantities we need on the surfaces we need.
                interp = interp1d(s_transform, iota_transform, fill_value="extrapolate")
                iota = interp(self.surfaces)

        # TODO: replace with current at different radial points
        # TODO: Why does spec.poloidal_current_amperes expist, isn't curpol already in Ampere?
        G = np.ones_like(self.surfaces) * spec.inputlist.curtor  # Should be Ampere
        I = np.ones_like(self.surfaces) * spec.inputlist.curpol  # Should be Ampere

        myshape = (ns, ntheta, nphi)
        assert modB.shape == myshape
        assert sqrtg.shape == myshape
        assert d_B_d_theta.shape == myshape
        assert d_B_d_phi.shape == myshape
        assert bsubu.shape == myshape
        assert bsubv.shape == myshape
        assert bsupu.shape == myshape
        assert bsupv.shape == myshape

        # From here on, the code is copied from vmec_diagnostics.py QuasisymmetryRatioResidual
        residuals3d = np.zeros(myshape)
        B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
        B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg

        dtheta = theta1d[1] - theta1d[0]
        dphi = phi1d[1] - phi1d[0]
        V_prime = nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2)) * ds
        # Check that we can evaluate the flux surface average <1> and the result is 1:
        assert np.sum(np.abs(np.sqrt((1 / V_prime) * nfp * dtheta * dphi * np.sum(sqrtg * ds[:,None,None], axis=(1, 2))) - 1)) < 1e-12

        nn = self.helicity_n * nfp
        for js in range(ns):
            residuals3d[js, :, :] = np.sqrt(self.weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js, :, :] * ds[js,None,None]) \
                * (B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * self.helicity_m) \
                   - B_dot_grad_B[js, :, :] * (self.helicity_m * G[js] + nn * I[js])) \
                / (modB[js, :, :] ** 3)
        

        import matplotlib.pyplot as plt
        plotvariables = {
            # "modB": modB,
            # "d_B_d_theta": d_B_d_theta,
            # "d_B_d_phi": d_B_d_phi,
            # "bsubv": bsubv,
            # "bsupv": bsupv,
            "sqrtg inner": sqrtg[0],
            "sqrtg inner+1": sqrtg[1], 
            "sqrtg outer": sqrtg[-1], 
            "+term": B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * self.helicity_m),
            "-term": B_dot_grad_B[js, :, :] * (self.helicity_m * G[js] + nn * I[js]),
            # "sqrtg3": sqrtg[32],
            # "sqrtg4": sqrtg[48],
            "residuals3d": residuals3d[-1]
        }
        print("d_psi_d_s SPEC", d_psi_d_s) 
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 6))
        for key, val, ax in zip(plotvariables.keys(), plotvariables.values(), axs.flatten()):
            pcm = ax.imshow(val, origin="lower")
            ax.set_title(key)
            fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        plt.text(0.5, 0.95, "SPEC", horizontalalignment="center", verticalalignment="center")
        plt.suptitle("SPEC")
        # plt.show()
        
        from scipy import integrate
        print("spec volume from sqrtg", nfp * integrate.simpson( y=integrate.simpson( y=integrate.simpson( y=sqrtg, x=phi1d ), x=theta1d ) * ds, x=self.surfaces ))
        vols = np.array([spec.results.get_volume(ivol, ns=32) for ivol in range(spec.nvol)]) 
        print("Volume Using py_spec",sum(vols), vols)


        if np.isnan(residuals3d).any():
            raise ObjectiveFailure("QuasisymmetryRatioResidualSpec failed")
        
        residuals1d = residuals3d.reshape((ns * ntheta * nphi,))
        profile = np.sum(residuals3d * residuals3d, axis=(1, 2))
        total = np.sum(residuals1d * residuals1d)


        # Form a structure with all the intermediate data as attributes:
        results = Struct()
        variables = ['ns', 'ntheta', 'nphi', 'dtheta', 'dphi', 'nfp', 'V_prime', 'theta1d', 'phi1d',
                     'd_psi_d_s', 'B_dot_grad_B',
                     'B_cross_grad_B_dot_grad_psi', 'modB', 'd_B_d_theta', 'd_B_d_phi', 'sqrtg',
                     'bsubu', 'bsubv', 'bsupu', 'bsupv', 'G', 'I', 'iota',
                     'residuals3d', 'residuals1d', 'profile', 'total']
        for v in variables:
            results.__setattr__(v, eval(v))

        logger.debug('Done evaluating quasisymmetry residuals')
        return results

    def residuals(self):
        """
        Evaluate the quasisymmetry metric in terms of a 1D numpy vector of
        residuals, corresponding to :math:`R` in the documentation
        for this class. This is the function to use when forming a
        least-squares objective function.
        """
        results = self.compute()
        return results.residuals1d

    def profile(self):
        """
        Return the quasisymmetry metric in terms of a 1D radial
        profile. The residuals :math:`R` are squared and summed over
        theta and phi, but not over s. The total quasisymmetry error
        :math:`f` returned by the :func:`total()` function is the sum
        of the values in the profile returned by this function.
        """
        results = self.compute()
        return results.profile

    def total(self):
        """
        Evaluate the quasisymmetry metric in terms of the scalar total
        :math:`f`.
        """
        results = self.compute()
        return results.total
