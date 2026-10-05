"""Сверка lab1.py со scipy.optimize.linprog на 1500 случайных ЗЛП."""
import random, io, contextlib, os, tempfile
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
import lab1

def run(c, mode, A, ops, b, free=()):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(' '.join(map(str, c)) + '\n' + mode + '\n')
        for r, o, bi in zip(A, ops, b):
            f.write(' '.join(map(str, r)) + f' {o} {bi}\n')
        if free: f.write('free ' + ' '.join(str(k+1) for k in free) + '\n')
        p = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = lab1.solve(p)
    os.unlink(p)
    return res

def ref(c, mode, A, ops, b, free=()):
    n = len(c)
    cc = np.array(c, float) * (1 if mode == 'min' else -1)
    Aub, bub, Aeq, beq = [], [], [], []
    for r, o, bi in zip(A, ops, b):
        if o == '<=': Aub.append(r); bub.append(bi)
        elif o == '>=': Aub.append([-v for v in r]); bub.append(-bi)
        else: Aeq.append(r); beq.append(bi)
    bounds = [(None, None) if j in free else (0, None) for j in range(n)]
    r = linprog(cc, A_ub=Aub or None, b_ub=bub or None, A_eq=Aeq or None, b_eq=beq or None, bounds=bounds, method='highs')
    return r

# lecture examples
tests = [
 ([2,1,-1],'min',[[3,0,2],[1,-1,1],[1,0,1]],['>=','=','<='],[12,5,6],()),            # slide 13-16, ans (0,1,6) W=-5
 ([3,2,5,1],'max',[[2,1,3,2],[1,4,2,3],[3,1,1,4]],['=','=','='],[8,10,9],()),        # infeasible example
 ([2,3,-1,6],'max',[[1,3,0,-1],[2,0,6,4],[0,1,-1,1]],['>=','=','>='],[-7,5,2],(2,)),  # free x3, negative b
 ([1,2,3,1],'min',[[1,2,1,0],[0,1,1,1],[1,0,0,1]],['<=','=','>='],[7,6,2],()),        # variant 2
 ([1,1],'max',[[1,-1]],['<='],[2],()),                                                # unbounded
 ([1,1],'min',[[1,1],[1,1]],['>=','<='],[10,2],()),                                   # infeasible
]
for t in tests:
    r = run(*t); rf = ref(*t)
    print('status ref:', rf.status, '| mine:', None if r is None else ([fmt for fmt in map(str, r[0])], str(r[1])), '| ref obj', None if rf.status else (rf.fun if t[1]=='min' else -rf.fun))

# random tests
random.seed(1)
bad = 0; cnt = {'opt':0,'inf':0,'unb':0}
for it in range(1500):
    n = random.randint(2,5); m = random.randint(1,4)
    c = [random.randint(-5,5) for _ in range(n)]
    A = [[random.randint(-4,4) for _ in range(n)] for _ in range(m)]
    ops = [random.choice(['<=','>=','=']) for _ in range(m)]
    b = [random.randint(-8,12) for _ in range(m)]
    mode = random.choice(['min','max'])
    free = tuple(j for j in range(n) if random.random() < 0.3)
    rf = ref(c, mode, A, ops, b, free)
    r = run(c, mode, A, ops, b, free)
    if rf.status == 0:
        cnt['opt'] += 1
        obj = rf.fun if mode == 'min' else -rf.fun
        if r is None or abs(float(r[1]) - obj) > 1e-6:
            bad += 1; print('MISMATCH opt', c, mode, A, ops, b, free, r, obj)
        else:
            x = r[0]   # feasibility check of my x
            for row, o, bi in zip(A, ops, b):
                lhs = sum(Fr(a)*xi for a, xi in zip(row, x))
                assert (o=='<=' and lhs<=bi) or (o=='>=' and lhs>=bi) or (o=='=' and lhs==bi)
            assert all(x[j] >= 0 for j in range(n) if j not in free)
    elif rf.status == 2:
        cnt['inf'] += 1
        if r is not None: bad += 1; print('MISMATCH should be infeasible', c, mode, A, ops, b, free)
    elif rf.status == 3:
        cnt['unb'] += 1
        if r is not None: bad += 1; print('MISMATCH should be unbounded', c, mode, A, ops, b, free)
    else:
        print('ref other status', rf.status)
print('bad =', bad, cnt)
