# twi_fast_valid.py
# Fast, valid TWI implementation (D8) with proper sink fill, flat routing, and topo accumulation.
import numpy as np
from numba import njit, prange

# ---------------------------
# D8 neighbourhood (N, NE, E, SE, S, SW, W, NW)
# ---------------------------
D8 = np.array([
    (-1,  0),  # N
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
], dtype=np.int8)

@njit(cache=True)
def _d8_distances(cell: float) -> np.ndarray:
    out = np.empty(8, np.float32)
    for k in range(8):
        dy, dx = int(D8[k,0]), int(D8[k,1])
        out[k] = np.sqrt((dy*cell)*(dy*cell) + (dx*cell)*(dx*cell))
    return out

# ---------------------------
# Priority-flood depression filling (Numba heap)
# ---------------------------
@njit(cache=True)
def _heap_push(h_e, h_y, h_x, size, e, y, x):
    i = size
    h_e[i] = e; h_y[i] = y; h_x[i] = x
    size += 1
    while i > 0:
        p = (i - 1) // 2
        if h_e[p] <= h_e[i]:
            break
        # swap
        te=h_e[p]; ty=h_y[p]; tx=h_x[p]
        h_e[p]=h_e[i]; h_y[p]=h_y[i]; h_x[p]=h_x[i]
        h_e[i]=te;    h_y[i]=ty;    h_x[i]=tx
        i = p
    return size

@njit(cache=True)
def _heap_pop(h_e, h_y, h_x, size):
    e = h_e[0]; y = h_y[0]; x = h_x[0]
    size -= 1
    h_e[0] = h_e[size]; h_y[0] = h_y[size]; h_x[0] = h_x[size]
    i = 0
    while True:
        l = 2*i + 1
        if l >= size: break
        r = l + 1
        m = l
        if r < size and h_e[r] < h_e[l]:
            m = r
        if h_e[i] <= h_e[m]:
            break
        te=h_e[i]; ty=h_y[i]; tx=h_x[i]
        h_e[i]=h_e[m]; h_y[i]=h_y[m]; h_x[i]=h_x[m]
        h_e[m]=te;    h_y[m]=ty;    h_x[m]=tx
        i = m
    return e, y, x, size

@njit(cache=True)
def priority_flood_fill_numba(dem: np.ndarray) -> np.ndarray:
    """
    Barnes-style priority-flood: raises depressions to spill height.
    Treats NaNs as barriers; preserves them.
    """
    z = dem.copy()
    h, w = z.shape
    valid = np.isfinite(z)

    visited = np.zeros((h,w), np.uint8)
    inq     = np.zeros((h,w), np.uint8)

    # over-allocate heap arrays to n cells
    n = h*w
    heap_e = np.empty(n, np.float32)
    heap_y = np.empty(n, np.int32)
    heap_x = np.empty(n, np.int32)
    size = 0

    # seed boundary
    for x in range(w):
        if valid[0,x] and inq[0,x]==0:
            size = _heap_push(heap_e,heap_y,heap_x,size,z[0,x],0,x); inq[0,x]=1
        if valid[h-1,x] and inq[h-1,x]==0:
            size = _heap_push(heap_e,heap_y,heap_x,size,z[h-1,x],h-1,x); inq[h-1,x]=1
    for y in range(1,h-1):
        if valid[y,0] and inq[y,0]==0:
            size = _heap_push(heap_e,heap_y,heap_x,size,z[y,0],y,0); inq[y,0]=1
        if valid[y,w-1] and inq[y,w-1]==0:
            size = _heap_push(heap_e,heap_y,heap_x,size,z[y,w-1],y,w-1); inq[y,w-1]=1

    while size > 0:
        elev, y, x, size = _heap_pop(heap_e,heap_y,heap_x,size)
        if visited[y,x]==1: 
            continue
        visited[y,x]=1
        for k in range(8):
            ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
            if ny<0 or ny>=h or nx<0 or nx>=w: 
                continue
            if valid[ny,nx]==0 or visited[ny,nx]==1: 
                continue
            if z[ny,nx] < elev:
                z[ny,nx] = elev
            if inq[ny,nx]==0:
                size = _heap_push(heap_e,heap_y,heap_x,size,z[ny,nx],ny,nx)
                inq[ny,nx]=1

    # preserve NaNs
    for i in range(h):
        for j in range(w):
            if not np.isfinite(dem[i,j]):
                z[i,j] = np.nan
    return z

# ---------------------------
# Horn slope -> tan(beta) (vectorised)
# ---------------------------
def horn_tan_slope(dem: np.ndarray, cell: float) -> np.ndarray:
    z = dem
    h, w = z.shape
    zp = np.pad(z, ((1,1),(1,1)), mode="edge").astype(np.float32)

    z1 = zp[0:h,   0:w  ]; z2 = zp[0:h,   1:w+1]; z3 = zp[0:h,   2:w+2]
    z4 = zp[1:h+1, 0:w  ]; z5 = zp[1:h+1, 1:w+1]; z6 = zp[1:h+1, 2:w+2]
    z7 = zp[2:h+2, 0:w  ]; z8 = zp[2:h+2, 1:w+1]; z9 = zp[2:h+2, 2:w+2]

    dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * cell)
    dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * cell)
    tanb = np.hypot(dzdx, dzdy).astype(np.float32)

    tanb[~np.isfinite(z5)] = np.nan
    return tanb

# ---------------------------
# D8 flow directions (steepest descent using true distances)
# ---------------------------
@njit(cache=True, fastmath=True)
def d8_flow_dirs(filled: np.ndarray, cell: float) -> np.ndarray:
    h, w = filled.shape
    dirs = np.full((h,w), -1, np.int8)
    dists = _d8_distances(cell)
    for y in range(h):
        for x in range(w):
            zc = filled[y,x]
            if not np.isfinite(zc):
                dirs[y,x] = -1
                continue
            best = -1e30; bi = -1
            for k in range(8):
                ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
                if ny<0 or ny>=h or nx<0 or nx>=w: 
                    continue
                zn = filled[ny,nx]
                if not np.isfinite(zn):
                    continue
                drop = zc - zn
                if drop <= 0.0:
                    continue
                s = drop / dists[k]
                if s > best:
                    best = s; bi = k
            dirs[y,x] = bi
    return dirs

# ---------------------------
# Flat routing across equal-height plateaus (BFS)
# ---------------------------
@njit(cache=True)
def resolve_flats_bfs(filled: np.ndarray, dirs: np.ndarray, tol: float = 0.0) -> np.ndarray:
    h, w = filled.shape
    dist = np.full((h,w), -1, np.int32)

    def eq(a,b):
        return np.abs(a-b) <= tol

    qy = np.empty(h*w, np.int32); qx = np.empty(h*w, np.int32)
    head = 0; tail = 0

    # seeds: flat-edge cells (touch lower + equal)
    for y in range(h):
        for x in range(w):
            zc = filled[y,x]
            if not np.isfinite(zc): 
                continue
            has_eq = False; has_lower = False
            for k in range(8):
                ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
                if ny<0 or ny>=h or nx<0 or nx>=w: 
                    continue
                zn = filled[ny,nx]
                if not np.isfinite(zn): 
                    continue
                if eq(zn, zc): has_eq = True
                elif zn < zc:  has_lower = True
            if has_eq and has_lower:
                dist[y,x] = 0
                qy[tail] = y; qx[tail] = x; tail += 1

    # BFS across equal-height cells
    while head < tail:
        y = qy[head]; x = qx[head]; head += 1
        zc = filled[y,x]; d = dist[y,x]
        for k in range(8):
            ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
            if ny<0 or ny>=h or nx<0 or nx>=w: 
                continue
            if dist[ny,nx] != -1: 
                continue
            zn = filled[ny,nx]
            if np.isfinite(zn) and eq(zn, zc):
                dist[ny,nx] = d + 1
                qy[tail] = ny; qx[tail] = nx; tail += 1

    out = dirs.copy()
    # assign directions in plateau interiors (dirs==-1) towards decreasing dist
    for y in range(h):
        for x in range(w):
            if out[y,x] != -1:
                continue
            zc = filled[y,x]
            if not np.isfinite(zc):
                continue
            # check we are in a plateau (has equal neighbours)
            has_eq = False
            for k in range(8):
                ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
                if ny<0 or ny>=h or nx<0 or nx>=w: 
                    continue
                zn = filled[ny,nx]
                if np.isfinite(zn) and eq(zn, zc):
                    has_eq = True; break
            if not has_eq:
                continue
            bestk = -1; bestd = 1<<30
            for k in range(8):
                ny = y + int(D8[k,0]); nx = x + int(D8[k,1])
                if ny<0 or ny>=h or nx<0 or nx>=w: 
                    continue
                zn = filled[ny,nx]
                if not np.isfinite(zn) or not eq(zn, zc):
                    continue
                dn = dist[ny,nx]
                if dn >= 0 and dn < bestd:
                    bestd = dn; bestk = k
            out[y,x] = bestk  # may remain -1 on isolated flat islands (rare after fill)
    return out

# ---------------------------
# Flow accumulation (topological, Kahn queue)
# ---------------------------
@njit(cache=True, fastmath=True)
def flow_accumulation_d8(dirs: np.ndarray) -> np.ndarray:
    h, w = dirs.shape
    n = h*w
    acc = np.ones(n, np.float64)
    offs = (D8[:,0].astype(np.int32)*w + D8[:,1].astype(np.int32))
    indeg = np.zeros(n, np.int32)
    down = np.full(n, -1, np.int32)

    fd = dirs.ravel()
    for i in range(n):
        d = fd[i]
        if d >= 0:
            y = i // w; x = i - y*w
            ny = y + D8[d,0]; nx = x + D8[d,1]
            if 0 <= ny < h and 0 <= nx < w:
                j = i + offs[d]
                down[i] = j
                indeg[j] += 1

    q = np.empty(n, np.int32); head=0; tail=0
    for i in range(n):
        if indeg[i]==0:
            q[tail]=i; tail+=1

    while head<tail:
        u = q[head]; head+=1
        v = down[u]
        if v >= 0:
            acc[v] += acc[u]
            indeg[v] -= 1
            if indeg[v]==0:
                q[tail]=v; tail+=1

    return acc.reshape((h,w))

# ---------------------------
# End-to-end TWI (arrays)
# ---------------------------
def twi_from_array(
    dem: np.ndarray,
    cellsize: float,
    slope_eps: float = 1e-6,
    do_fill: bool = True,
    flats_tol: float = 0.0
) -> np.ndarray:
    """
    Compute TWI = ln(a / tanβ) with D8 routing.
    - dem: float32 array, NaN = nodata
    - a = accumulation_cells * cellsize  (specific catchment area, m² per m)
    """
    dem = np.ascontiguousarray(dem.astype(np.float32))
    filled = priority_flood_fill_numba(dem) if do_fill else dem

    tanb = horn_tan_slope(filled, float(cellsize))
    dirs = d8_flow_dirs(filled, float(cellsize))
    dirs = resolve_flats_bfs(filled, dirs, tol=float(flats_tol))
    acc_cells = flow_accumulation_d8(dirs).astype(np.float32)

    a = acc_cells * float(cellsize)
    twi = np.log(a / (np.maximum(tanb, float(slope_eps)))).astype(np.float32)

    # honour nodata
    twi[~np.isfinite(dem)] = np.nan
    return twi

# ---------------------------
# Raster I/O convenience
# ---------------------------
def compute_twi_raster(
    in_path: str,
    out_path: str,
    slope_eps: float = 1e-6,
    do_fill: bool = True,
    flats_tol: float = 0.0,
    dtype: str = "float32"
):
    try:
        import rasterio
    except Exception as e:
        raise RuntimeError("rasterio is required for raster I/O") from e

    with rasterio.open(in_path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            dem = np.where(dem == src.nodata, np.nan, dem)
        cell = float(abs(src.transform.a))
        profile = src.profile

    twi = twi_from_array(dem, cellsize=cell, slope_eps=slope_eps, do_fill=do_fill, flats_tol=flats_tol)

    profile.update(dtype=dtype, count=1, nodata=np.nan, compress="LZW")
    import rasterio
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(twi.astype(dtype), 1)
