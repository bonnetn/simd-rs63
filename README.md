# SIMD Reed Solomon

A Rust implementation of Reed–Solomon RS(6,3) using SIMD instructions for fast GF(256) arithmetic. 

It encodes 6 data shards into 3 parity shards, producing a 9-shard codeword, and can recover the original data when any 0 to 3 shards are missing.

It uses **no unsafe code** while still compiling down to SIMD instructions, thanks to [Rust's portable SIMD API](https://doc.rust-lang.org/std/simd/index.html). It has **zero dependencies**.

On a 2020 MacBook Air M1, it reaches **22 GiB/s/thread** for encoding parity and **42 GiB/s/thread** when recovering a single missing shard.

## The math behind it

### Notations

The idea is that, given 6 bytes of a message:

```math
\text{message} = \mathbf{m} = \begin{bmatrix} m_0 & m_1 & m_2 & m_3 & m_4 & m_5  \end{bmatrix}^{\mathsf T}
```

We'd like to compute 3 bytes of parity
```math
\text{parity} = \mathbf{p} = \begin{bmatrix} p_0 & p_1 & p_2\end{bmatrix}^{\mathsf T}
```

These 9 bytes form a codeword :
```math
\text{codeword} = \mathbf{c} = \begin{bmatrix} m_0 & m_1 & m_2 & m_3 & m_4 & m_5 & p_0 & p_1 & p_2 \end{bmatrix}^{\mathsf T}
```


The idea is that, given you have the value and position of 6 distinct bytes from the 9-byte codeword, you can always recover the other 3.

$G$ is the matrix that maps the message to the codeword:
```math
\begin{aligned}
\mathbf{c} = G\,\mathbf{m} 
\quad \text{with} \quad
G \in \mathbb{GF}(256)^{9 \times 6}
\qquad{(2)}
\end{aligned}
```

H is the parity-check matrix of the code, that verifies:
```math
H\,\mathbf{c} = \mathbf{0}_{3 \times 1}
\quad \text{with} \quad
H \in \mathbb{GF}(256)^{3 \times 9}
\qquad{(3)}
```


### How to find missing bytes?

Assume that we have only 6 of the 9 bytes of a codeword.

We call the bytes we have "surviving" bytes, and the bytes we lost "missing". 

$\mathbf{c}_s$ is a column vector containing 6 of the 9 bytes from the original codeword $\mathbf{c}$.


From (2):
```math
\mathbf{c} = G\,\mathbf{m} \qquad{(2)}
```

Selecting only the rows from $\mathbf{c}$ of the "survivor" bytes:
```math
\begin{aligned}
\mathbf{c}_s &= G_s \mathbf{m}
\quad \text{with} \quad
\mathbf{c}_s \in \mathbb{GF}(256)^{6 \times 1},
\quad 
G_s \in \mathbb{GF}(256)^{6 \times 6} \\
\mathbf{m} &= G_s ^ {-1} \mathbf{c}_s
\end{aligned}
```

Plugging it back into (2)
```math
\begin{aligned}
\Aboxed{\mathbf{c} = (G G_s ^ {-1}) \mathbf{c}_s} \\
\end{aligned}
```

For every choice of survivor positions, we can compute the matrix $G G_s ^ {-1}$ that will allow us to recover the full codeword. 

Recovering the missing bytes just boils down to multiplying matrices.

> [!NOTE]
> We only need to keep/use the rows for the missing values.


### How do we compute G?

Top part of G must be the identity matrix because of (1)  .
```math
G = 
\begin{aligned}
\begin{bmatrix}
I_6 \\
P
\end{bmatrix} \qquad{(4)} \\

\text{with} \quad P \in \mathbb{GF}(256)^{3 \times 6}
\end{aligned}
```


Given (1), (2), (4):
```math
\begin{aligned}
\mathbf{c} &= \begin{bmatrix}
\mathbf{m} \\
P\mathbf{m}
\end{bmatrix} \\
\mathbf{p} &= P \mathbf{m} \\
\end{aligned}
```

We can split $H$
```math
\begin{aligned}
H = 
\begin{bmatrix}
H_m & H_p
\end{bmatrix} \\
\text{with} \quad H_m \in \mathbb{GF}(256)^{3 \times 6},
\quad H_p \in \mathbb{GF}(256)^{3 \times 3}
\end{aligned}
```

So that (from (1) + (3)):
```math
H_m \mathbf{m} + H_p \mathbf{p} = \mathbf{0}
```


Then we find $P$:
```math
\begin{aligned}
\mathbf{p} = -H_p ^ {-1} H_m \mathbf{m} \\

\Aboxed{
P = -H_p ^ {-1} H_m
}
\end{aligned}

\qquad{(5)}
```

With $P$ we trivially get $G$ using (4).


### How do we compute H?

Let $f(x)$ be:
```math
f(x) = c_0 + c_1 x + c_2 x^2 + \cdots + c_{n-1} x^{n-1}
```

(With $c_i$ the bytes of the codeword $\mathbf{c}$)

We arbitrarily pick $\alpha^i$ as roots of $f(x)$.

$\alpha$ is a primitive element of $\mathbb{GF}(2^8)$.

So we have, $i \in \{1,2,3\}$:
```math
f(root_i) = f(\alpha^i) = c_0 + c_1 \alpha^i + c_2 \alpha^{2i} + ... + c_{n-1} \alpha^{(n-1)i}
```

Or in matrix form:
```math
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
=
\begin{bmatrix}
1 & \alpha & \alpha^2 & \cdots & \alpha^{n-1} \\
1 & \alpha ^ 2 & \alpha^4 & \cdots & \alpha^{2(n-1)} \\
1 & \alpha ^ 3 & \alpha^6 & \cdots & \alpha^{3(n-1)} \\
\end{bmatrix}
\begin{bmatrix}
c_0 \\
c_1 \\
\vdots \\
c_{n-1}
\end{bmatrix}
```

We want to find a matrix so that:
```math
H\,\mathbf{c} = \mathbf{0}_{3 \times 1} \qquad{(3)}
```

Notice that the matrix we just found satifies (3).

So we choose
```math
H = 
\begin{bmatrix}
1 & \alpha & \alpha^2 & \cdots & \alpha^{n-1} \\
1 & \alpha ^ 2 & \alpha^4 & \cdots & \alpha^{2(n-1)} \\
1 & \alpha ^ 3 & \alpha^6 & \cdots & \alpha^{3(n-1)} \\
\end{bmatrix}
```

With:
```math
H_{i,j} = \alpha ^ {i(j-1)}
\qquad
1 \le i \le 3,\ 1 \le j \le n
```

Which is simply a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix) over $\mathbb{GF}_{256}$.

### Putting it all together

We know how to compute $H$...
which allows us to compute $P$...
which allows us to compute $G$...
which allows us $G G_s ^ {-1}$ for all combinaisons of error positions.

This program has to simply pre-compute all matrices $\text{FixMatrix} = G G_s ^ {-1}$, and when trying to fix a errors, do a matrix multiplication $\text{FixMatrix} \cdot \mathbf{c}_s$.

$G G_s ^ {-1}$ easy!! 
