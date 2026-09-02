# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 22:18:13
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-09 11:57:07
from numpy._typing._array_like import NDArray
import torch as tc
import numpy as np
device = "cuda" if tc.cuda.is_available() else "cpu"
from snqs import *
from utils import *
from rbm import *
from model import *
from sampler import *
from vmc import *
import math
from model import *



tim = TIM(J=1.0, hx=1.0, hz=1.0)
bonds = tim.bonds(Lx=4, Ly=1)
print(bonds)
