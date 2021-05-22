sens = 0.9
spec = 1.0 

pop_slum    = 4202
pop_nonslum = 2702 
pop = pop_slum + pop_nonslum

S = pop_slum    / pop 
N = pop_nonslum / pop

sero_slum    = 58.4
sero_nonslum = 17.3
sero = S * sero_slum + N * sero_nonslum

var_slum    = abs(sero_slum    - 56.8)
var_nonslum = abs(sero_nonslum - 18.7)

var = (
    S**2 * var_slum    +\
    N**2 * var_nonslum
)**0.5

print([round(_, 2) for _ in (sero, sero - var, sero + var)])