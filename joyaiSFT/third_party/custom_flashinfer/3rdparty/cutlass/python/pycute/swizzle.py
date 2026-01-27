"""
Methods for layout swizzling
"""
from .layout import *

def shiftr(a, s):
    return a >> s if s > 0 else shiftl(a, -s)

def shiftl(a, s):
    return a << s if s > 0 else shiftr(a, -s)

class Swizzle:

    def __init__(self, bits, base, shift):
        assert bits >= 0
        assert base >= 0
        assert abs(shift) >= bits
        self.bits = bits
        self.base = base
        self.shift = shift
        bit_msk = (1 << bits) - 1
        self.yyy_msk = bit_msk << base + max(0, shift)
        self.zzz_msk = bit_msk << base - min(0, shift)

    def __call__(self, offset):
        return offset ^ shiftr(offset & self.yyy_msk, self.shift)

    def size(self):
        return 1 << bits + base + abs(shift)

    def cosize(self):
        return self.size()

    def __str__(self):
        return f'SW_{self.bits}_{self.base}_{self.shift}'

    def __repr__(self):
        return f'Swizzle({self.bits},{self.base},{self.shift})'

class ComposedLayout(LayoutBase):

    def __init__(self, layoutB, offset, layoutA):
        self.layoutB = layoutB
        self.offset = offset
        self.layoutA = layoutA

    def __eq__(self, other):
        return self.layoutB == other.layoutB and self.offset == other.offset and (self.layoutA == other.layoutA)

    def __len__(self):
        return len(self.layoutA)

    def __call__(self, *args):
        return self.layoutB(self.offset + self.layoutA(*args))

    def __getitem__(self, i):
        return ComposedLayout(self.layoutB, self.offset, self.layoutA[i])

    def size(self):
        return size(self.layoutA)

    def cosize(self):
        return cosize(self.layoutB)

    def __str__(self):
        return f'{self.layoutB} o {self.offset} o {self.layoutA}'

    def __repr__(self):
        return f'ComposedLayout({repr(self.layoutB)},{repr(self.offset)},{repr(self.layoutA)})'