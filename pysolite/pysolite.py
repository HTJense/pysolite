"""
	SO CMB-only likelihood. Based on MFLike.
	Written by Hidde Jense.
"""

import os, sys
import sacc
from typing import Optional
import numpy as np
from scipy import linalg
from cobaya.likelihood import Likelihood

class SOCMBLikelihood(Likelihood):
	input_file: Optional[str]
	defaults: dict
	
	def initialize(self):
		self.l_bpws = None
		self.freqs = None
		self.spec_meta = [ ]
		self.data_vec = None
		self.inv_cov = None
		
		data_file_path = os.path.normpath(getattr(self, "path", None) or os.path.join(self.packages_path, "data"))
		self.data_folder = os.path.join(data_file_path, self.data_folder)
		
		if not os.path.exists(self.data_folder):
			raise LoggedError(self.log, "Data folder [{self.data_folder}] does not exist.")
		
		self.prepare_data()
		
		self.lmax_theory = 9000
		self.requested_cls = [ "tt", "te", "ee" ]
		
		self.log.info("Initialization done.")
	
	def get_requirements(self):
		return dict( Cl = { k : self.lmax_theory + 1 for k in self.requested_cls } )
	
	def prepare_data(self):
		input_fname = "sacc_cmbext.fits"
		s = sacc.Sacc.load_fits(input_fname)
		
		indices = []
		index_sofar = 0
		
		for pol in self.defaults['polarizations']:
			p1, p2 = pol.lower()
			if p1 == 't': p1 = '0'
			if p2 == 't':
				p1, p2 = '0', p1
			
			dt = 'cl_' + p1 + p2
		
			if not dt in s.get_data_types():
				raise LoggedError(f'Requested data type {dt} not present in sacc file.')
			
			combinations = s.get_tracer_combinations(dt)
			if len(combinations) == 0:
				raise LoggedError(f'Data type {dt} does not have any tracer combinations in sacc file.')
			
			lmin, lmax = self.defaults['scales'][pol]
			
			for f1, f2 in combinations:
				ind = s.indices(dt, f1, f2, ell__gt = lmin, ell__lt = lmax)
				ls, cls = s.get_ell_cl(dt, f1, f2)
				indices += list(ind)
				
				ws = s.get_data_points(dt, (f1, f2))[0].get_tag("window")
				
				if self.l_bpws is None:
					self.l_bpws = ws.values
				
				self.spec_meta.append({
					"ind" : index_sofar + np.arange(cls.size, dtype = int),
					"pol" : pol.lower(),
					"leff" : ls,
					"cl_data" : cls,
					"bpw" : ws
				})
				
				index_sofar += cls.size
				
				if self.data_vec is None:
					self.data_vec = cls
				else:
					self.data_vec = np.concatenate((self.data_vec, cls))
		
		self.cov = s.covariance.covmat
		self.inv_cov = np.linalg.inv(self.cov)
		
		self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2)
		self.logp_const -= 0.5 * np.linalg.slogdet(self.cov)[1]
	
	def logp(self, **params_values):
		cl = self.theory.get_Cl(ell_factor = True)
		return self.loglike(cl)
	
	def loglike(self, cl):
		ps_vec = self.get_power_spectra(cl)
		delta = self.data_vec - ps_vec
		logp = -0.5 * (delta @ self.inv_cov @ delta)
		logp += self.logp_const
		return logp
	
	def get_power_spectra(self, cl):
		dls = { s : cl[s][self.l_bpws] for s in ["tt", "te", "ee"] }
		ps_vec = np.zeros_like(self.data_vec)
		
		for m in self.spec_meta:
			pol = m["pol"]
			ind = m["ind"]
			ws = m["bpw"].weight.T
			ps_vec[ind] = np.dot(ws, dls[pol])
		
		return ps_vec
