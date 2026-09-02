# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-12 15:07:21
# @Last Modified by:   dzwang
# @Last Modified time: 2026-02-03 19:17:06
import torch as tc

 
class TIM:
    
    def __init__(self, J:float, hx:float, hz:float) -> None:
        self.J = J
        self.hx = hx
        self.hz = hz
    
    def bonds(self, Lx:int, Ly:int=1) -> tc.Tensor:
        bonds = []        
        # Vertical bonds (connect sites in same column)
        for x in range(Lx):
            for y in range(Ly - 1):
                site1 = x * Ly + y
                site2 = x * Ly + y + 1
                bonds.append([site1, site2])
        # Horizontal bonds (connect sites in same row) 
        for x in range(Lx - 1):
            for y in range(Ly):
                site1 = x * Ly + y
                site2 = (x + 1) * Ly + y
                bonds.append([site1, site2])
        return tc.tensor(bonds)

