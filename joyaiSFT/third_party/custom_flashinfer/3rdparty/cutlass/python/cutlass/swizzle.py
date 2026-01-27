"""
Registry of swizzling functions
"""
from cutlass_library import SwizzlingFunctor
IdentitySwizzle1 = SwizzlingFunctor.Identity1
IdentitySwizzle2 = SwizzlingFunctor.Identity2
IdentitySwizzle4 = SwizzlingFunctor.Identity4
IdentitySwizzle8 = SwizzlingFunctor.Identity8
HorizontalSwizzle = SwizzlingFunctor.Horizontal
ThreadblockSwizzleStreamK = SwizzlingFunctor.StreamK
StridedDgradIdentitySwizzle1 = SwizzlingFunctor.StridedDgradIdentity1
StridedDgradIdentitySwizzle4 = SwizzlingFunctor.StridedDgradIdentity4
StridedDgradHorizontalSwizzle = SwizzlingFunctor.StridedDgradHorizontal
_swizzling_functors = [IdentitySwizzle1, IdentitySwizzle2, IdentitySwizzle4, IdentitySwizzle8, HorizontalSwizzle, ThreadblockSwizzleStreamK, StridedDgradIdentitySwizzle1, StridedDgradIdentitySwizzle4, StridedDgradHorizontalSwizzle]

def get_swizzling_functors():
    return _swizzling_functors