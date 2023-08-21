NLay
=====

**N**Pad **Lay**outing Library

NLay (pronounced "Enlay") is layouting library inspired by the flexibility of Android's [ConstraintLayout](https://developer.android.com/training/constraint-layout).
This layouting library attempts to implement subset of the ConstraintLayout layouting functionality.

NLay is **NOT** full UI library. It merely function as helper on element placement on the screen.

This Python library is 1:1 mapping with its [Lua equivalent](https://github.com/MikuAuahDark/NPad93#nlay), based on NLay 1.4. It is also PEP-484 compliant.

Example
-----

```py
# Please don't use `from nlay import *`. The module object itself will be used!
import nlay

# Yes you look at it right. The `nlay` module variable is used.
rect = nlay.inside(nlay, 10).constraint(nlay, nlay, nlay, nlay)
testgrid = nlay.grid(nlay, 4, 4, spacing=4.0, spacingfl=True)

print(rect.get()) # Print the are where to draw the rect.
```

More information about the the function documentation can be found here: https://github.com/MikuAuahDark/NPad93#nlay

TODO
-----

* Integration with Tkinter
