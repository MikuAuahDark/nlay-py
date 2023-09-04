import dataclasses as dataclasses
import fractions as fractions
import math as math
import weakref as weakref

from typing import Generic, Literal, Protocol, TypeVar


class BaseConstraint(Protocol):
    def get(self, offx: float = 0.0, offy: float = 0.0) -> tuple[float, float, float, float]:
        ...


UnitMode = Literal["percent"] | Literal["pixel"]
LineDirection = Literal["horizontal"] | Literal["vertical"]

root_x = 0
root_y = 0
root_width = 800
root_height = 600


@dataclasses.dataclass
class _Cache:
    result: tuple[float, float, float, float] | None = None
    referenced: weakref.WeakSet[BaseConstraint] = dataclasses.field(default_factory=weakref.WeakSet)


cache: weakref.WeakKeyDictionary[BaseConstraint, _Cache] = weakref.WeakKeyDictionary()


def get_cached_data(constraint: BaseConstraint):
    global cache
    result = cache.get(constraint)
    return None if result is None else result.result


def get_cache_entry(constraint: BaseConstraint):
    global cache
    c = cache.get(constraint)

    if c is None:
        c = _Cache()
        cache[constraint] = c

    return c


def insert_cached(constraint: BaseConstraint, values: tuple[float, float, float, float]):
    c = get_cache_entry(constraint)
    c.result = values


def invalidate_cache(constraint: BaseConstraint):
    c = get_cache_entry(constraint)

    # Constraint that reference `constraint` must be invalidated too.
    if c.result:
        for k in c.referenced:
            invalidate_cache(k)

        c.result = None


def add_ref_cache(constraint: BaseConstraint, other: BaseConstraint):
    c = get_cache_entry(constraint)
    c.referenced.add(other)


def lerp(a: float, b: float, t: float):
    return a * (1.0 - t) + b * t


_T = TypeVar("_T")


def pick(*args: _T | None) -> _T:
    for k in args:
        if k is not None:
            return k
    raise TypeError("all type is none")


def clamp01(f: float):
    return min(max(float(f), 0.0), 1.0)


_U = TypeVar("_U")


class Constraint(Generic[_U]):
    def __init__(
        self,
        ins: "Inside",
        top: BaseConstraint | None,
        left: BaseConstraint | None,
        bottom: BaseConstraint | None,
        right: BaseConstraint | None,
    ) -> None:
        self.ins = ins
        self.top, self.left, self.right, self.bottom = top, left, bottom, right
        self.in_top, self.in_left, self.in_bottom, self.in_right = False, False, False, False
        self.into_flagged_by_user = False
        self.margin_x, self.margin_y, self.margin_w, self.margin_h = 0.0, 0.0, 0.0, 0.0
        self.w, self.h = -1.0, -1.0
        self.rel_w, self.rel_h = False, False
        self.bias_h, self.bias_v = 0.5, 0.5
        self.aspect_ratio: fractions.Fraction | None = None
        self.user_tag: _U | None = None
        self._override_into_flags()

    def _override_into_flags(self):
        self.in_top = self.in_top or self.ins.source() == self.top
        self.in_left = self.in_left or self.ins.source() == self.left
        self.in_bottom = self.in_bottom or self.ins.source() == self.bottom
        self.in_right = self.in_right or self.ins.source() == self.right

        if self.top == self.bottom and self.top is not None:
            self.in_top, self.in_bottom = True, True

        if self.left == self.right and self.left is not None:
            self.in_left, self.in_right = True, True

    def get(self, offx: float = 0.0, offy: float = 0.0):
        finals = get_cached_data(self)

        if finals is None:
            if (self.left is not None or self.right is not None) and (self.top is not None or self.bottom is not None):
                width, height = self.w, self.h
                zerodim = False

                # Convert percent values to pixel values
                if self.rel_w:
                    width = self._resolve_width_size_0()[1] * self.w

                if self.rel_h:
                    height = self._resolve_height_size_0()[1] * self.h

                # Resolve aspect ratio, part 1
                if self.aspect_ratio is not None:
                    if width == 0.0 and height != 0.0:
                        width = height * self.aspect_ratio
                    elif width != 0.0 and height == 0.0:
                        height = height / self.aspect_ratio
                    else:
                        zerodim = width == 0.0 and height == 0.0

                if zerodim:
                    resolved_width, resolved_height = None, None
                    assert self.aspect_ratio is not None

                    if self.left is not None and self.right is not None:
                        resolved_width = self._resolve_width_size_0()[1]

                    if self.top is not None and self.bottom is not None:
                        resolved_height = self._resolve_height_size_0()[1]

                    if resolved_width is not None or resolved_height is not None:
                        if resolved_width is not None and resolved_height is not None:
                            if (resolved_width / resolved_height) > self.aspect_ratio:
                                # h / ratio, h
                                height = resolved_width

                                if self.aspect_ratio.numerator > self.aspect_ratio.denominator:
                                    width = resolved_height * self.aspect_ratio
                                else:
                                    width = resolved_height / self.aspect_ratio
                            else:
                                width = resolved_width

                                if self.aspect_ratio.numerator > self.aspect_ratio.denominator:
                                    height = resolved_width * self.aspect_ratio
                                else:
                                    height = resolved_width / self.aspect_ratio
                        elif resolved_width is not None:
                            width, height = resolved_width, resolved_width
                        elif resolved_height is not None:
                            width, height = resolved_height, resolved_height

                # All data gathered. Now resolve!
                if width == -1.0:
                    # Match parent
                    p = self.ins.get()
                    x, width = p[0], p[2]
                elif width == 0.0:
                    x, width = self._resolve_width_size_0()

                if height == -1.0:
                    # Match parent
                    p = self.ins.get()
                    y, height = p[1], p[3]
                elif height == 0.0:
                    y, height = self._resolve_height_size_0()

                # Resolve aspect ratio, part 2
                if self.aspect_ratio is not None and zerodim:
                    maxw, maxh = width, height
                    cw, ch = height * self.aspect_ratio, width * self.aspect_ratio

                    if cw > maxw:
                        height = ch
                    elif ch > maxh:
                        width = cw

                # Resolve horizontal
                l, r, w = None, None, width

                if self.left is not None:
                    # Left orientation
                    e1 = self._resolve_constraint_size(self.left)
                    l = e1[0] + e1[2] * (not self.in_left) + self.margin_x

                if self.right is not None:
                    # Right orientation
                    e2 = self._resolve_constraint_size(self.right)
                    r = e2[0] + e2[2] * self.in_right - self.margin_w - w

                x = lerp(l, r, self.bias_h) if (l is not None and r is not None) else pick(l, r)

                # Resolve vertical
                t, b, h = None, None, height

                if self.top is not None:
                    # Top orientation
                    e1 = self._resolve_constraint_size(self.top)
                    t = e1[1] + e1[3] * (not self.in_top) + self.margin_y

                if self.bottom is not None:
                    # Bottom orientation
                    e2 = self._resolve_constraint_size(self.bottom)
                    b = e2[1] + e2[3] * self.in_bottom - self.margin_h - h

                y = lerp(t, b, self.bias_v) if (t is not None and b is not None) else pick(t, b)

                finals = (x, y, max(w, 0.0), max(h, 0.0))
                insert_cached(self, finals)
            else:
                raise RuntimeError("insufficient constraint")

        return finals[0] + offx, finals[1] + offy, finals[2], finals[3]

    def into(self, top: bool = False, left: bool = False, bottom: bool = False, right: bool = False):
        """
        This function tells that for constraint specified at `top`, `left`, `bottom`, and/or `right`, it should NOT
        use the opposite sides of the constraint. This mainly used to prevent ambiguity.
        """

        self.in_top, self.in_left, self.in_bottom, self.in_right = top, left, bottom, right
        if not self.into_flagged_by_user:
            self._override_into_flags()

        invalidate_cache(self)
        return self

    def margin(
        self,
        margin: float | tuple[float | int | None, float | int | None, float | int | None, float | int | None] = 0.0,
    ):
        """
        Sets the constraint margin.
        """

        if isinstance(margin, float):
            m = float(margin)
            self.margin_x, self.margin_y, self.margin_w, self.margin_h = m, m, m, m
        elif isinstance(margin, tuple):
            self.margin_x, self.margin_y, self.margin_w, self.margin_h = tuple(
                float(margin[i] or 0.0) for i in range(4)
            )
        else:
            raise TypeError("expected float or tuple of 4 floats")

        invalidate_cache(self)
        return self

    def size(self, width: float, height: float, mode_w: UnitMode = "pixel", mode_h: UnitMode = "pixel"):
        """
        Sets the constraint width and height.

        If `width` or `height` is 0, it will calculate them based on the other connected constraint. If it's -1, then
        it will use parent's `width`/`height` minus padding. Otherwise it will try to place the constraint based on
        the bias (default 0.5).
        """

        width = float(width)
        if width != -1 and width < 0:
            raise ValueError("invalid width")

        height = float(height)
        if height != -1 and height < 0:
            raise ValueError("invalid height")

        self.w, self.h = width, height
        self.rel_w, self.rel_h = mode_w == "percent", mode_h == "percent"

        invalidate_cache(self)
        return self

    def bias(self, horz: float | None = None, vert: float | None = None, unclamped: bool = False):
        """
        Set the constraint bias.

        By default, for fixed width/height, the bias is 0.5 which means the position are centered.
        """

        op = float if unclamped else clamp01

        if horz is not None:
            self.bias_h = op(horz)

        if vert is not None:
            self.bias_v = op(vert)

        invalidate_cache(self)
        return self

    def force_in(self, force: bool = False):
        """
        Force the "into" flags to be determined by user even if it may result as invalid constraint. By default some
        "into" flags were determined automatically. Setting this function to true causes NLay not to determine the
        "into" flags automatically. This function is only used for some "niche" cases. You don't have to use this
        almost all the time.
        """
        self.into_flagged_by_user = force
        return self

    def tag(self, tag: _U | None):
        """
        Tag this constraint with user-specific data (i.e. id). Useful to keep track of constraints when they're
        rebuilt.
        """

        self.user_tag = tag
        return self

    def get_tag(self):
        """
        Retrieve tag data from constraint (or `None` if this constraint is not tagged). See above function for more
        information.
        """

        return self.user_tag

    def ratio(self, ratio: float | tuple[int, int] | fractions.Fraction | None = None):
        """
        Set the size aspect ratio. The `ratio` can be tuple of 2 ints, a float in `numerator/denominator`, or a
        `Fraction` object. So for aspect ratio of 16:9, pass `(16, 9)`, `16/9`, or `Fraction(16, 9)`.
        """

        if ratio is None:
            r = None
        elif isinstance(ratio, float):
            if math.isnan(ratio) or math.isinf(ratio) or ratio == 0.0:
                r = None
            else:
                r = abs(fractions.Fraction.from_float(ratio))
        elif isinstance(ratio, tuple):
            r = fractions.Fraction(*ratio)
        elif isinstance(ratio, fractions.Fraction):
            r = ratio
        else:
            raise TypeError("expected float, tuple of 2 ints, Fraction, or None")

        if r is not None:
            r = abs(r)
            if r == 0:
                r = None

        self.aspect_ratio = r
        invalidate_cache(self)
        return self

    def _resolve_width_size_0(self):
        if self.left is None or self.right is None:
            raise RuntimeError("insufficient constraint for width 0")

        # Left
        e = self._resolve_constraint_size(self.left)
        x = e[0] + e[2] * self.in_left + self.margin_x
        # Right
        e = self._resolve_constraint_size(self.right)
        width = e[0] + e[2] * self.in_right - x - self.margin_w

        return x, width

    def _resolve_height_size_0(self):
        if self.top is None or self.bottom is None:
            raise RuntimeError("insufficient constraint for height 0")

        # Top
        e = self._resolve_constraint_size(self.top)
        y = e[1] + e[3] * self.in_top + self.margin_y
        # Bottom
        e = self._resolve_constraint_size(self.bottom)
        height = e[1] + e[3] * self.in_bottom - y - self.margin_y

        return y, height

    def _resolve_constraint_size(self, target: BaseConstraint):
        if target == self.ins.source():
            return self.ins.get()
        else:
            return target.get()


class Inside:
    """
    Create new `Inside` object. This object is used to construct `Constraint` later on.
    """

    def __init__(
        self,
        constraint: BaseConstraint,
        padding: float | tuple[float | int | None, float | int | None, float | int | None, float | int | None] = 0.0,
    ) -> None:
        self.obj = constraint
        if isinstance(padding, (float, int)):
            p = float(padding)
            self.padding = p, p, p, p
        elif isinstance(padding, tuple):
            self.padding = tuple(float(padding[i] or 0.0) for i in range(4))
        else:
            raise TypeError("expected float or tuple of 4 floats")

    def constraint(
        self,
        top: BaseConstraint | None,
        left: BaseConstraint | None,
        bottom: BaseConstraint | None = None,
        right: BaseConstraint | None = None,
    ):
        result = Constraint(self, top, left, bottom, right)

        if top is not None:
            add_ref_cache(result, top)
        if left is not None:
            add_ref_cache(result, left)
        if bottom is not None:
            add_ref_cache(result, bottom)
        if right is not None:
            add_ref_cache(result, right)

        return result

    def source(self):
        return self.obj

    def get(self):
        x, y, w, h = self.obj.get()
        return (
            x + self.padding[1],
            y + self.padding[0],
            w - self.padding[3] - self.padding[1],
            h - self.padding[2] - self.padding[0],
        )


class MaxConstraint:
    """
    Create new constraint whose the size and the position is based on bounding box of the other constraint. At least 2
    constraint must be passed to this function.
    """

    def __init__(self, constraint: BaseConstraint, *constraints: BaseConstraint) -> None:
        self.constraints = (constraint, *constraints)

        for c in self.constraints:
            add_ref_cache(self, c)

    def get(self, offx: float = 0.0, offy: float = 0.0):
        minx, miny, maxx, maxy = math.inf, math.inf, -math.inf, -math.inf

        for c in self.constraints:
            x, y, w, h = c.get()
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + h)

        return minx + offx, miny + offy, maxx - minx, maxx - miny


class LineConstraint:
    """
    Create new [guideline constraint](https://developer.android.com/training/constraint-layout#constrain-to-a-guideline).

    Direction can be either "horizontal" or "vertical". Horizontal direction creates vertical line with width of 0 for
    constraint to attach horizontally. Vertical direction creates horizontal line with height of 0 for constraint to
    attach vertically.

    Mode can be either "percent" or "pixel". If it's percentage, then value is bias inside the constraint where 0
    denotes top/left and 1 denotes bottom/right. If it's pixel, then it behaves identical to "margin". Negative values
    start the bias/offset from opposing direction.
    """

    def __init__(
        self, constraint: BaseConstraint | Inside, direction: LineDirection, mode: UnitMode, offset: float
    ) -> None:
        self.constraint = constraint.source() if isinstance(constraint, Inside) else constraint
        self.direction = direction
        self.mode_op = LineConstraint.percent_mode if mode == "percent" else LineConstraint.pixel_mode
        self.line_offset = offset
        self.flip = 1 / offset < 0 if offset != 0.0 else False
        add_ref_cache(self, self.constraint)

    def get(self, offx: float = 0.0, offy: float = 0.0):
        x, y, w, h = self.constraint.get()
        finals = get_cached_data(self)

        if finals is None:
            if self.direction == "horizontal":
                finals = self.mode_op(x, self.flip, w, self.line_offset), y, 0.0, h
            else:
                finals = x, self.mode_op(y, self.flip, h, self.line_offset), w, 0.0

            insert_cached(self, finals)

        return finals[0] + offx, finals[1] + offy, finals[2], finals[3]

    def offset(self, off: float):
        self.line_offset = float(off)
        invalidate_cache(self)
        return self

    @staticmethod
    def pixel_mode(v: float, flip: bool, a: float, offset: float):
        return v + (a if flip else 0.0) + offset

    @staticmethod
    def percent_mode(v: float, flip: bool, a: float, offset: float):
        return lerp(v, v + a, (1.0 if flip else 0.0) + offset)


class GridCellConstraint:
    def __init__(self, context: "Grid", tx: int, ty: int) -> None:
        self.context = context
        self.tx, self.ty = tx, ty

    def get(self, offx: float = 0.0, offy: float = 0.0):
        x, y, w, h = self.context.resolve_cell(self.tx, self.ty)
        return x + offx, y + offy, w, h


class GridCellIterator:
    def __init__(self, context: "Grid") -> None:
        self.context = context
        self.len = len(self.context.list)
        self.index = 0

    def __next__(self):
        if self.index >= self.len:
            raise StopIteration

        # row * self.cols + col
        row = self.index // self.context.cols
        col = self.index % self.context.cols
        self.index = self.index + 1
        return row, col, self.context.get(row, col)


class Grid:
    def __init__(
        self,
        constraint: BaseConstraint | Constraint,
        nrows: int,
        ncols: int,
        *,
        hspacing: float | None = None,
        vspacing: float | None = None,
        spacing: float = 0.0,
        hspacingfl: bool | None = None,
        vspacingfl: bool | None = None,
        spacingfl: bool = False,
        cellwidth: float = 0.0,
        cellheight: float = 0.0,
    ) -> None:
        if nrows < 1:
            raise ValueError("n rows out of range")
        if ncols < 1:
            raise ValueError("n cols out of range")

        self.vspacing = pick(vspacing, spacing, 0.0)
        self.hspacing = pick(hspacing, spacing, 0.0)
        if self.vspacing < 0.0:
            raise ValueError("vertical spacing out of range")
        if self.hspacing < 0.0:
            raise ValueError("horizontal spacing out of range")
        self.vfl = pick(vspacingfl, spacingfl, False)
        self.hfl = pick(hspacingfl, spacingfl, False)

        if cellwidth < 0.0:
            raise ValueError("cell width out of range")
        if cellheight < 0.0:
            raise ValueError("cell height out of range")
        if (cellwidth > 0.0 and cellheight == 0.0) or (cellwidth == 0.0 and cellheight > 0.0):
            raise ValueError("need cell width and cell height")

        self.constraint = constraint
        self.cell_w = cellwidth
        self.cell_h = cellheight
        self.rows = nrows
        self.cols = ncols
        self.list: list[GridCellConstraint | None] = [None] * (nrows * ncols)

        if cellwidth > 0.0 and cellheight > 0.0:
            if not isinstance(constraint, Constraint):
                raise ValueError("fixed cell requires Constraint object")
            self.update_size()

    def get(self, row: int, col: int):
        index = col * self.rows + row
        constraint = self.list[index]

        if constraint is None:
            constraint = GridCellConstraint(self, col, row)
            self.list[index] = constraint

        return constraint

    def spacing(self, h: float | None = None, v: float | None = None, hfl: bool | None = None, vfl: bool | None = None):
        self.hspacing = pick(h, self.hspacing)
        self.vspacing = pick(v, self.vspacing)
        self.hfl = pick(hfl, self.hfl)
        self.vfl = pick(vfl, self.vfl)
        return self

    def cellsize(self, width: float, height: float):
        if self.is_fixed:
            if width <= 0.0:
                raise ValueError("cell width out of range")
            if height <= 0.0:
                raise ValueError("cell height out of range")

            self.cell_w, self.cell_h = width, height
            self.update_size()

        return self

    @property
    def is_fixed(self):
        return self.cell_w > 0.0 and self.cell_h > 0.0

    def get_cell_dimensions(self):
        return self.resolve_cell(0, 0)[2:]

    def update_size(self):
        assert isinstance(self.constraint, Constraint)
        width = self.hspacing * (self.cols + (self.hfl * 2 - 1)) + self.cell_w * self.cols
        height = self.vspacing * (self.rows + (self.vfl * 2 - 1)) + self.cell_h * self.rows
        self.constraint.size(width, height)

    def resolve_cell(self, x: int, y: int):
        xc, yc, w, h = self.constraint.get()
        if self.is_fixed:
            w, h = self.cell_w, self.cell_h
        else:
            cell_w = (w - self.hspacing * (self.cols + (self.hfl * 2 - 1))) / self.cols
            cell_h = (h - self.vspacing * (self.rows + (self.vfl * 2 - 1))) / self.rows
            w, h = cell_w, cell_h

        xp = (x + self.hfl) * self.hspacing + x * w
        yp = (y + self.vfl) * self.vspacing + y * h
        return xp + xc, yp + yc, w, h

    def __iter__(self):
        return GridCellIterator(self)


def get(offx: float = 0, offy: float = 0):
    global root_x, root_y, root_width, root_height
    return float(root_x) + offx, float(root_y) + offy, float(root_width), float(root_height)


def update(x: int, y: int, width: int, height: int):
    """
    Update the window dimensions. In most cases, you want `x` and `y` to be 0.
    """

    global root_x, root_y, root_width, root_height
    root_x, root_y, root_width, root_height = x, y, width, height
