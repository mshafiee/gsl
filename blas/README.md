# BLAS Support in Go

The Basic Linear Algebra Subprograms (BLAS) define a set of fundamental
operations on vectors and matrices which can be used to create optimized
higher-level linear algebra functionality.

This package provides a low-level layer which corresponds to Go-style
BLAS operations, It is suggested to use these package, rather than
going for other BLAS libraries because:
  - They were implemented in pure Go which means, no CGO or Assembly calls.
  - Most of them uses go-routines, which are way faster than other languages threads
  - They have been implemented keeping the *Go Idioms* in mind

Note that GSL matrices are implemented using dense-storage so the
interface only includes the corresponding dense-storage BLAS
functions.  The full BLAS functionality for band-format and
packed-format matrices is available through the low-level CBLAS
interface.  Similarly, GSL vectors are restricted to positive strides.
 
The complete set of BLAS functions is
listed in an appendix.

There are three levels of BLAS operations,

| Level   | Description                                        |
| :------ | :------------------------------------------------- |
| Level 1 | Vector operations, e.g.  `y = αx + y`             |
| Level 2 | Matrix-vector operations, e.g.  `y = αAx + βy`   |
| Level 3 | Matrix-matrix operations, e.g.  `C = αAB + C`     |

Each routine has a name which specifies the operation, the type of
matrices involved and their precisions.  Some of the most common
operations and their names are given below,

| Name  | Description               |
| :---- | :------------------------ |
| DOT   | scalar product,  `x^T y`  |
| AXPY  | vector sum,  `αx + y`      |
| MV    | matrix-vector product,  `A x`  |
| SV    | matrix-vector solve,  `inv(A) x` |
| MM    | matrix-matrix product,  `A B`  |
| SM    | matrix-matrix solve,  `inv(A) B` |

The types of matrices are,

| Type | Description         |
| :--- | :------------------ |
| GE   | general             |
| GB   | general band        |
| SY   | symmetric           |
| SB   | symmetric band      |
| SP   | symmetric packed    |
| HE   | hermitian           |
| HB   | hermitian band      |
| HP   | hermitian packed    |
| TR   | triangular          |
| TB   | triangular band     |
| TP   | triangular packed   |

Each operation is defined for four precisions,

| Prefix | Description      |
| :----- | :--------------- |
| S      | single real      |
| D      | double real      |
| C      | single complex   |
| Z      | double complex   |

Thus, for example, the name SGEMM stands for "single-precision
general matrix-matrix multiply" and ZGEMM stands for
"double-precision complex matrix-matrix multiply".

Note that the vector and matrix arguments to BLAS functions must not
be aliased, as the results are undefined when the underlying arrays
overlap.

## GSL BLAS Interface

GSL provides dense vector and matrix objects, based on the relevant
built-in types.  The library provides an interface to the BLAS
operations which apply to these objects.  The interface to this
functionality is given in the file `blas.go`.

## Level 1

### func Sdsdot

```go
func Sdsdot(alpha float32, x *VectorFloat, y *VectorFloat) (float32, error)
```

Sdsdot computes the dot product of two float32 vectors with single precision accumulation, plus a scalar: `α + x^T y`.  Returns an error if the vector sizes are unequal.

### func Dsdot

```go
func Dsdot(x *VectorFloat, y *VectorFloat) (float64, error)
```
Dsdot computes the dot product of two float32 vectors with double precision accumulation: `x^T y`. Returns an error if the vector sizes are unequal.

### func Sdot

```go
func Sdot(x *VectorFloat, y *VectorFloat) (float32, error)
```
Sdot computes the dot product of two float32 vectors: `x^T y`. Returns an error if the vector sizes are unequal.

### func Ddot

```go
func Ddot(x *Vector, y *Vector) (float64, error)
```

Ddot computes the dot product of two float64 vectors: `x^T y`. Returns an error if the vector sizes are unequal.

### func Cdotu

```go
func Cdotu(x *VectorComplexFloat, y *VectorComplexFloat) (GslComplexFloat, error)
```

Cdotu computes the dot product of two complex vectors (unconjugated): `x^T y`. Returns an error if the vector sizes are unequal.

### func Cdotc

```go
func Cdotc(x *VectorComplexFloat, y *VectorComplexFloat) (GslComplexFloat, error)
```

Cdotc computes the dot product of two complex vectors (conjugated): `x^H y`. Returns an error if the vector sizes are unequal.

### func Zdotu

```go
func Zdotu(x *VectorComplex, y *VectorComplex) (GslComplex, error)
```

Zdotu computes the dot product of two complex vectors (unconjugated): `x^T y`. Returns an error if the vector sizes are unequal.

### func Zdotc

```go
func Zdotc(x *VectorComplex, y *VectorComplex) (GslComplex, error)
```

Zdotc computes the dot product of two complex vectors (conjugated): `x^H y`.  Returns an error if the vector sizes are unequal.

### func Snrm2

```go
func Snrm2(x *VectorFloat) float32
```

Snrm2 computes the Euclidean norm of a float32 vector: `||x||_2 = sqrt(Σ x_i^2)`.

### func Sasum

```go
func Sasum(x *VectorFloat) float32
```

Sasum computes the sum of absolute values of a float32 vector: `Σ |x_i|`.

### func Dnrm2

```go
func Dnrm2(x *Vector) float64
```

Dnrm2 computes the Euclidean norm of a float64 vector:  `||x||_2 = sqrt(Σ x_i^2)`.

### func Dasum

```go
func Dasum(x *Vector) float64
```

Dasum computes the sum of absolute values of a float64 vector: `Σ |x_i|`.

### func Scnrm2

```go
func Scnrm2(x *VectorComplexFloat) float32
```

Scnrm2 computes the Euclidean norm of a complex vector: `||x||_2 = sqrt(Σ (Re(x_i)^2 + Im(x_i)^2))`.

### func Scasum

```go
func Scasum(x *VectorComplexFloat) float32
```

Scasum computes the sum of the magnitudes of the real and imaginary parts of a complex vector: `Σ (|Re(x_i)| + |Im(x_i)|)`.

### func Dznrm2

```go
func Dznrm2(x *VectorComplex) float64
```

Dznrm2 computes the Euclidean norm of a complex vector:`||x||_2 = sqrt(Σ (Re(x_i)^2 + Im(x_i)^2))`.

### func Dzasum

```go
func Dzasum(x *VectorComplex) float64
```

Dzasum computes the sum of the magnitudes of the real and imaginary parts of a complex vector: `Σ (|Re(x_i)| + |Im(x_i)|)`.

### func Isamax

```go
func Isamax(x *VectorFloat) int
```

Isamax finds the index of the element with the largest absolute value in a float32 vector. If the largest value occurs several times, the index of the first occurrence is returned.  Returns 0 for an empty vector.

### func Idamax

```go
func Idamax(x *Vector) int
```

Idamax finds the index of the element with the largest absolute value in a float64 vector.  If the largest value occurs several times, the index of the first occurrence is returned. Returns 0 for an empty vector.

### func Icamax

```go
func Icamax(x *VectorComplexFloat) int
```

Icamax finds the index of the element with the largest absolute value (sum of magnitudes of real and imaginary parts) in a complex vector. If the largest value occurs several times, the index of the first occurrence is returned. Returns 0 for an empty vector.

### func Izamax

```go
func Izamax(x *VectorComplex) int
```

Izamax finds the index of the element with the largest absolute value (sum of magnitudes of real and imaginary parts) in a complex vector.  If the largest value occurs several times, the index of the first occurrence is returned. Returns 0 for an empty vector.

### func Sswap

```go
func Sswap(x, y *VectorFloat) error
```

Sswap swaps two float32 vectors. Returns an error if the vector sizes are unequal.

### func Scopy

```go
func Scopy(x, y *VectorFloat) error
```

Scopy copies a float32 vector to another float32 vector. Returns an error if the vector sizes are unequal.

### func Saxpy

```go
func Saxpy(alpha float32, x *VectorFloat, y *VectorFloat) error
```

Saxpy computes `y = alpha * x + y` for float32 vectors. Returns an error if the vector sizes are unequal.

### func Dswap

```go
func Dswap(x, y *Vector) error
```

Dswap swaps two float64 vectors. Returns an error if the vector sizes are unequal.

### func Dcopy

```go
func Dcopy(x, y *Vector) error
```

Dcopy copies a float64 vector to another float64 vector. Returns an error if the vector sizes are unequal.

### func Daxpy

```go
func Daxpy(alpha float64, x *Vector, y *Vector) error
```

Daxpy computes `y = alpha * x + y` for float64 vectors. Returns an error if the vector sizes are unequal.

### func Cswap

```go
func Cswap(x, y *VectorComplexFloat) error
```
Cswap swaps two complex float32 vectors. Returns an error if the vector sizes are unequal.

### func Ccopy

```go
func Ccopy(x, y *VectorComplexFloat) error
```
Ccopy copies content of complex float32 vector to another complex float32 vector.

### func Caxpy

```go
func Caxpy(alpha GslComplexFloat, x, y *VectorComplexFloat) error
```

Caxpy computes `y = alpha * x + y` for complex float32 vectors. Returns an error if the vector sizes are unequal.

### func Zswap

```go
func Zswap(x, y *VectorComplex) error
```

Zswap swaps two complex float64 vectors. Returns an error if the vector sizes are unequal.

### func Zcopy

```go
func Zcopy(x, y *VectorComplex) error
```
Zcopy copies a complex float64 vector to another complex vector. Returns an error if vector sizes are not equal.

### func Zaxpy

```go
func Zaxpy(alpha GslComplex, x, y *VectorComplex) error
```

Zaxpy computes y = alpha * x + y for complex float64 vectors. Returns an error if the vector sizes are unequal.

### func Srotg

```go
func Srotg(a, b float32) (c, s, r, z float32)
```

Srotg generates a float32 Givens rotation.

### func Srot

```go
func Srot(x, y *VectorFloat, c, s float32) error
```

Srot applies a float32 Givens rotation to two float32 vectors. Returns an error if the vector sizes are unequal.

### func Srotmg

```go
func Srotmg(d1, d2, b1, b2 *float32) (p [5]float32)
```

Srotmg generates a modified float32 Givens rotation.

### func Srotm

```go
func Srotm(x, y *VectorFloat, p [5]float32) error
```

Srotm applies a modified Givens rotation to two float32 vectors. Returns an error if the vector sizes are unequal.

### func Drotg

```go
func Drotg(a, b float64) (c, s, r, z float64)
```

Drotg generates a Givens rotation.

### func Drot

```go
func Drot(x, y *Vector, c, s float64) error
```

Drot applies a Givens rotation to two float64 vectors. Returns an error if the vector sizes are unequal.

### func Drotmg

```go
func Drotmg(d1, d2, b1, b2 *float64) (p [5]float64)
```

Drotmg generates a modified Givens rotation.

### func Drotm

```go
func Drotm(x, y *Vector, p [5]float64) error
```

Drotm applies a modified Givens rotation to two float64 vectors. Returns an error if the vector sizes are unequal.

### func Sscal

```go
func Sscal(alpha float32, x *VectorFloat)
```

Sscal scales a float32 vector by a float32 scalar: `x = alpha * x`.

### func Dscal

```go
func Dscal(alpha float64, x *Vector)
```

Dscal scales a float64 vector by a float64 scalar: `x = alpha * x`.

### func Cscal

```go
func Cscal(alpha GslComplexFloat, x *VectorComplexFloat)
```

Cscal scales a complex vector by a complex scalar.

### func Zscal

```go
func Zscal(alpha GslComplex, x *VectorComplex)
```

Zscal scales a complex vector by a complex scalar.

### func Csscal

```go
func Csscal(alpha float32, x *VectorComplexFloat)
```

Csscal scales a complex vector by a float32 scalar.

### func Zdscal

```go
func Zdscal(alpha float64, x *VectorComplex)
```

Zdscal scales a complex vector by a float64 scalar.

## Level 2

### func Sgemv
```go
func Sgemv(transA CblasTranspose, alpha float32, a *MatrixFloat, x *VectorFloat, beta float32, y *VectorFloat) error
```
Sgemv computes `y = alpha * op(A) * x + beta * y` (general matrix-vector multiplication), where `op(A)` is `A`, `A^T`, or `A^H` depending on `transA`.
Returns an error if the matrix and vector sizes are incompatible.

### func Strmv
```go
func Strmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixFloat, x *VectorFloat) error
```
Strmv computes `x = op(A) * x` for a triangular matrix `A`, where `op(A)` is `A`, `A^T`, or `A^H` depending on `transA`.
`uplo` specifies whether the upper or lower triangle of `A` is used.
`diag` specifies whether the diagonal of `A` is unit or not.
Returns an error if the matrix is not square or if vector and matrix sizes are incompatible.

### func Strsv
```go
func Strsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixFloat, x *VectorFloat) error
```
Strsv solves `op(A) * x = b` for `x`, where A is a triangular matrix and op(A) is A, A^T, or A^H for transA = CblasNoTrans, CblasTrans, CblasConjTrans.
When uplo is CblasUpper then the upper triangle of A is used, and when uplo is CblasLower then the lower triangle of A is used.
If diag is CblasNonUnit then the diagonal of the matrix is used, but if diag is CblasUnit then the diagonal elements of the matrix A are taken as unity and are not referenced.
Returns an error if the input matrix is not square or if the vector and matrix sizes are incompatible.

### func Dgemv
```go
func Dgemv(transA CblasTranspose, alpha float64, a *Matrix, x *Vector, beta float64, y *Vector) error
```

Dgemv computes `y = alpha * op(A) * x + beta * y` (general matrix-vector multiplication).
Returns an error if the matrix and vector sizes are incompatible.

### func Dtrmv

```go
func Dtrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *Matrix, x *Vector) error
```
Dtrmv computes `x = op(A) * x` for a triangular matrix `A`.
Returns an error if the matrix is not square or if vector and matrix sizes are incompatible.

### func Dtrsv
```go
func Dtrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *Matrix, x *Vector) error
```
Dtrsv solves `op(A) * x = b` for `x`, where `A` is a triangular matrix. Returns an error if the input matrix is not square or if the vector and matrix sizes are incompatible.

### func Cgemv

```go
func Cgemv(transA CblasTranspose, alpha GslComplexFloat, a *MatrixComplexFloat, x *VectorComplexFloat, beta GslComplexFloat, y *VectorComplexFloat) error
```
Cgemv computes `y = alpha * op(A) * x + beta * y` (general matrix-vector multiplication). Returns error if sizes are incompatible.

### func Ctrmv
```go
func Ctrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplexFloat, x *VectorComplexFloat) error
```

Ctrmv computes `x = op(A) * x` for a triangular matrix `A`.
Returns an error if the matrix is not square or if vector and matrix sizes are incompatible.

### func Ctrsv
```go
func Ctrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplexFloat, x *VectorComplexFloat) error
```
Ctrsv solves `op(A) * x = b` for `x`, where `A` is a triangular matrix. Returns an error if the matrix is not square or if vector and matrix sizes are incompatible.

### func Zgemv
```go
func Zgemv(transA CblasTranspose, alpha GslComplex, a *MatrixComplex, x *VectorComplex, beta GslComplex, y *VectorComplex) error
```
Zgemv computes `y = alpha * op(A) * x + beta * y`  (general matrix-vector multiplication). Returns an error if the sizes are incompatible.

### func Ztrmv

```go
func Ztrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplex, x *VectorComplex) error
```

Ztrmv computes `x = op(A) * x` for a triangular matrix `A`.
Returns an error if the matrix is not square or if vector and matrix sizes are incompatible.

### func Ztrsv

```go
func Ztrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplex, x *VectorComplex) error
```

Ztrsv solves `op(A) * x = b` for `x`, where `A` is a triangular matrix. Returns an error if the input matrix is not square of if the matrix and vector sizes are incompatible.

### func Sger
```go
func Sger(alpha float32, x *VectorFloat, y *VectorFloat, a *MatrixFloat) error
```

Sger computes `A = alpha * x * y^T + A` (rank-1 update).  Returns error if the vector and matrix sizes are incompatible.

### func Dger
```go
func Dger(alpha float64, x *Vector, y *Vector, a *Matrix) error
```
Dger computes `A = alpha * x * y^T + A` (rank-1 update). Returns error if the vector and matrix sizes are incompatible.

### func Cgeru
```go
func Cgeru(alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error
```
Cgeru computes `A = alpha * x * y^T + A` (rank-1 update, unconjugated). Returns error if the vector and matrix sizes are incompatible.

### func Zgeru
```go
func Zgeru(alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error
```
Zgeru computes `A = alpha * x * y^T + A` (rank-1 update, unconjugated).  Returns error if the vector and matrix sizes are incompatible.

### func Cgerc
```go
func Cgerc(alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error
```
Cgerc computes `A = alpha * x * y^H + A` (rank-1 update, conjugated). Returns error if the vector and matrix sizes are incompatible.

### func Zgerc
```go
func Zgerc(alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error
```
Zgerc computes `A = alpha * x * y^H + A` (rank-1 update, conjugated). Returns error if the vector and matrix sizes are incompatible.

### func Ssymv
```go
func Ssymv(uplo CblasUplo, alpha float32, a *MatrixFloat, x *VectorFloat, beta float32, y *VectorFloat) error
```

Ssymv performs the matrix-vector operation `y := alpha*A*x + beta*y`,
Returns error if the matrix is not square, or if vector lengths are not compatible with the matrix size.

### func Dsymv
```go
func Dsymv(uplo CblasUplo, alpha float64, a *Matrix, x *Vector, beta float64, y *Vector) error
```
Dsymv performs the matrix-vector operation  `y := alpha*A*x + beta*y`.
Returns error if the matrix is not square, or if vector lengths are not compatible with the matrix size.

### func Ssyr
```go
func Syr(uplo CblasUplo, alpha float32, x *VectorFloat, a *MatrixFloat) error
```

Ssyr computes `A = alpha * x * x^T + A` (symmetric rank-1 update). Returns an error if the input matrix is not square or if the vector and matrix sizes are incompatible.

### func Dsyr
```go
func Dsyr(uplo CblasUplo, alpha float64, x *Vector, a *Matrix) error
```
Dsyr computes `A = alpha * x * x^T + A` (symmetric rank-1 update). Returns an error if the input matrix is not square or if the vector and matrix sizes are incompatible.

### func Ssyr2

```go
func Syr2(uplo CblasUplo, alpha float32, x *VectorFloat, y *VectorFloat, a *MatrixFloat) error
```

Ssyr2 computes `A = alpha * x * y^T + alpha * y * x^T + A` (symmetric rank-2 update).  Returns error if the matrix is not square or the vector and matrix sizes are incompatible.

### func Dsyr2
```go
func Dsyr2(uplo CblasUplo, alpha float64, x *Vector, y *Vector, a *Matrix) error
```
Dsyr2 computes `A = alpha * x * y^T + alpha * y * x^T + A` (symmetric rank-2 update). Returns an error if the matrix is not square or if the vector and matrix sizes are incompatible.

### func Chemv
```go
func Chemv(uplo CblasUplo, alpha GslComplexFloat, a *MatrixComplexFloat, x *VectorComplexFloat, beta GslComplexFloat, y *VectorComplexFloat) error
```
Chemv computes `y = alpha * A * x + beta * y` where A is hermitian (complex symmetric). Returns an error if the matrix is not square, of if vector lengths are not compatible with the matrix size.

### func Cher
```go
func Cher(uplo CblasUplo, alpha float32, x *VectorComplexFloat, a *MatrixComplexFloat) error
```
Cher computes `A = alpha * x * x^H + A` (Hermitian rank-1 update). Returns an error if the matrix is not square or if the vector and matrix sizes are incompatible.

### func Cher2
```go
func Cher2(uplo CblasUplo, alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error
```

Cher2 computes `A = alpha * x * y^H + conj(alpha) * y * x^H + A` (Hermitian rank-2 update).
Returns an error if the matrix is not square or if the vector and matrix sizes are incompatible.

### func Zhemv
```go
func Zhemv(uplo CblasUplo, alpha GslComplex, a *MatrixComplex, x *VectorComplex, beta GslComplex, y *VectorComplex) error
```
Zhemv computes `y = alpha * A * x + beta * y` where `A` is hermitian (complex symmetric). Returns an error if matrix is not square or if vector lengths are not compatible with matrix size.

### func Zher
```go
func Zher(uplo CblasUplo, alpha float64, x *VectorComplex, a *MatrixComplex) error
```
Zher computes `A = alpha * x * x^H + A` (Hermitian rank-1 update). Returns an error if matrix is not square or if the vector and matrix sizes are incompatible.

### func Zher2
```go
func Zher2(uplo CblasUplo, alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error
```
Zher2 computes `A = alpha * x * y^H + conj(alpha) * y * x^H + A` (Hermitian rank-2 update). Returns an error if matrix is not square or if the vector and matrix sizes are incompatible.

## Level 3

### func Sgemm

```go
func Sgemm(transA, transB CblasTranspose, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error
```

Sgemm performs one of the matrix-matrix operations `C := alpha*op( A )*op( B ) + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Dgemm

```go
func Dgemm(transA, transB CblasTranspose, alpha float64, a, b *Matrix, beta float64, c *Matrix) error
```

Dgemm performs one of the matrix-matrix operations `C := alpha*op( A )*op( B ) + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Cgemm

```go
func Cgemm(transA, transB CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error
```

Cgemm performs one of the matrix-matrix operations `C := alpha*op( A )*op( B ) + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Zgemm

```go
func Zgemm(transA, transB CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error
```
Zgemm performs one of the matrix-matrix operations `C := alpha*op( A )*op( B ) + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Ssymm
```go
func Ssymm(side CblasSide, uplo CblasUplo, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error
```
Ssymm performs one of the matrix-matrix operations   `C := alpha*A*B + beta*C` or  `C := alpha*B*A + beta*C`,
where `A` is a symmetric matrix. Returns an error if matrix dimensions are incompatible.

### func Dsymm
```go
func Dsymm(side CblasSide, uplo CblasUplo, alpha float64, a, b *Matrix, beta float64, c *Matrix) error
```

Dsymm performs one of the matrix-matrix operations   `C := alpha*A*B + beta*C` or  `C := alpha*B*A + beta*C`,
where `A` is a symmetric matrix. Returns an error if matrix dimensions are incompatible.

### func Csymm
```go
func Csymm(side CblasSide, uplo CblasUplo, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error

```
Csymm performs one of the matrix-matrix operations `C := alpha*A*B + beta*C` or `C := alpha*B*A + beta*C`
where `A` is symmetric matrix. Returns an error is matrix dimensions are incompatible.

### func Zsymm
```go
func Zsymm(side CblasSide, uplo CblasUplo, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error
```
Zsymm performs one of the matrix-matrix operations `C := alpha*A*B + beta*C` or `C := alpha*B*A + beta*C` where `A` is a symmetric matrix. Returns an error is the matrix dimensions are incompatible.

### func Ssyrk
```go
func Ssyrk(uplo CblasUplo, trans CblasTranspose, alpha float32, a *MatrixFloat, beta float32, c *MatrixFloat) error
```

Ssyrk performs one of the symmetric rank k operations `C := alpha*A*A**T + beta*C` or `C := alpha*A**T*A + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Dsyrk

```go
func Dsyrk(uplo CblasUplo, trans CblasTranspose, alpha float64, a *Matrix, beta float64, c *Matrix) error
```

Dsyrk performs one of the symmetric rank k operations `C := alpha*A*A**T + beta*C` or `C := alpha*A**T*A + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Csyrk
```go
func Csyrk(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error
```

Csyrk performs one of the symmetric rank k operations `C := alpha*A*A**T + beta*C` or `C := alpha*A**T*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

### func Zsyrk
```go
func Zsyrk(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a *MatrixComplex, beta GslComplex, c *MatrixComplex) error
```
Zsyrk performs one of the symmetric rank k operations `C := alpha*A*A**T + beta*C` or `C := alpha*A**T*A + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Ssyr2k
```go
func Ssyr2k(uplo CblasUplo, trans CblasTranspose, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error
```

Ssyr2k performs one of the symmetric rank 2k operations `C := alpha*A*B**T + alpha*B*A**T + beta*C` or `C := alpha*A**T*B + alpha*B**T*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

### func Dsyr2k

```go
func Dsyr2k(uplo CblasUplo, trans CblasTranspose, alpha float64, a, b *Matrix, beta float64, c *Matrix) error
```

Dsyr2k performs one of the symmetric rank 2k operations `C := alpha*A*B**T + alpha*B*A**T + beta*C` or `C := alpha*A**T*B + alpha*B**T*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

### func Csyr2k
```go
func Csyr2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error
```
Csyr2k performs one of the symmetric rank 2k operations `C := alpha*A*B**T + alpha*B*A**T + beta*C` or `C := alpha*A**T*B + alpha*B**T*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

###func Zsyr2k
```go
func Zsyr2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error
```
Zsyr2k performs one of the symmetric rank 2k operations `C := alpha*A*B**T + alpha*B*A**T + beta*C` or `C := alpha*A**T*B + alpha*B**T*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

### func Chemm
```go
func Chemm(side CblasSide, uplo CblasUplo, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error
```

Chemm performs one of the matrix-matrix operations `C := alpha*A*B + beta*C` or `C := alpha*B*A + beta*C`,
where `A` is an hermitian matrix.  Returns an error if matrix dimensions are incompatible.

###func Cherk
```go
func Cherk(uplo CblasUplo, trans CblasTranspose, alpha float32, a *MatrixComplexFloat, beta float32, c *MatrixComplexFloat) error
```
Cherk performs one of the hermitian rank k operations `C := alpha*A*A**H + beta*C` or `C := alpha*A**H*A + beta*C`.
Returns an error if matrix dimensions are incompatible.

### func Cher2k
```go
func Cher2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta float32, c *MatrixComplexFloat) error
```

Cher2k performs one of the hermitian rank 2k operations  `C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C`   or   `C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Zhemm

```go
func Zhemm(side CblasSide, uplo CblasUplo, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error
```

Zhemm performs one of the matrix-matrix operations `C := alpha*A*B + beta*C` or  `C := alpha*B*A + beta*C`, where `A` is an hermitian matrix. Returns an error if matrix dimensions are incompatible.

### func Zherk

```go
func Zherk(uplo CblasUplo, trans CblasTranspose, alpha float64, a *MatrixComplex, beta float64, c *MatrixComplex) error
```

Zherk performs one of the hermitian rank k operations `C := alpha*A*A**H + beta*C` or `C := alpha*A**H*A + beta*C`. Returns error if matrix dimensions are incompatible.

### func Zher2k
```go
func Zher2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta float64, c *MatrixComplex) error
```
Zher2k performs one of the hermitian rank 2k operations `C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C`   or   `C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C`. Returns an error if matrix dimensions are incompatible.

### func Strmm

```go
func Strmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float32, a *MatrixFloat, b *MatrixFloat) error
```
Strmm performs one of the matrix-matrix operations  `B := alpha*op( A )*B`, or `B := alpha*B*op( A )` where `op(A)` is `A`, `A^T`, or `A^H`. Returns an error if matrix dimensions are incompatible.

### func Dtrmm
```go
func Dtrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float64, a *Matrix, b *Matrix) error
```
Dtrmm performs one of the matrix-matrix operations  `B := alpha*op( A )*B`, or `B := alpha*B*op( A )` where `op(A)` is `A`, `A^T`, or `A^H`. Returns an error if matrix dimensions are incompatible.

### func Ctrmm

```go
func Ctrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplexFloat, a *MatrixComplexFloat, b *MatrixComplexFloat) error
```

Ctrmm performs one of the matrix-matrix operations `B := alpha*op( A )*B`, or `B := alpha*B*op( A )` where `op(A)` is `A`, `A^T`, or `A^H`. Returns an error if matrix dimensions are incompatible.

### func Ztrmm
```go
func Ztrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplex, a *MatrixComplex, b *MatrixComplex) error
```

Ztrmm performs one of the matrix-matrix operations `B := alpha*op( A )*B`, or `B := alpha*B*op( A )` where `op(A)` is `A`, `A^T`, or `A^H`. Returns an error if matrix dimensions are incompatible.

### func Strsm

```go
func Strsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float32, a *MatrixFloat, b *MatrixFloat) error
```

Strsm solves one of the matrix equations `op( A )*X = alpha*B`, or `X*op( A ) = alpha*B`
where `op(A)` is `A`, `A^T`, or `A^H`. Returns an error if matrix dimensions are incompatible.

### func Dtrsm

```go
func Dtrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float64, a *Matrix, b *Matrix) error
```
Dtrsm solves one of the matrix equations `op( A )*X = alpha*B, or X*op( A ) = alpha*B`
Returns an error if matrix dimensions are incompatible.

### func Ctrsm

```go
func Ctrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplexFloat, a *MatrixComplexFloat, b *MatrixComplexFloat) error
```
Ctrsm solves one of the matrix equations   `op( A )*X = alpha*B`, or `X*op( A ) = alpha*B`
Returns an error if matrix dimensions are incompatible.

###func Ztrsm
```go
func Ztrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplex, a *MatrixComplex, b *MatrixComplex) error
```
Ztrsm solves one of the matrix equations   `op( A )*X = alpha*B, or X*op( A ) = alpha*B`
Returns an error if matrix dimensions are incompatible.

## References and Further Reading

Information on the BLAS standards, including both the legacy and
updated interface standards, is available online from the BLAS
Homepage and BLAS Technical Forum web-site.

* BLAS Homepage, http://www.netlib.org/blas/

* BLAS Technical Forum, http://www.netlib.org/blas/blast-forum/

The following papers contain the specifications for Level 1, Level 2 and
Level 3 BLAS.

* C. Lawson, R. Hanson, D. Kincaid, F. Krogh, "Basic Linear Algebra
  Subprograms for Fortran Usage", ACM Transactions on Mathematical
  Software, Vol.: 5 (1979), Pages 308--325.

* J.J. Dongarra, J. DuCroz, S. Hammarling, R. Hanson, "An Extended Set of
  Fortran Basic Linear Algebra Subprograms", ACM Transactions on
  Mathematical Software, Vol.: 14, No.: 1 (1988), Pages 1--32.

* J.J. Dongarra, I. Duff, J. DuCroz, S. Hammarling, "A Set of
  Level 3 Basic Linear Algebra Subprograms", ACM Transactions on
  Mathematical Software, Vol.: 16 (1990), Pages 1--28.

Postscript versions of the latter two papers are available from
http://www.netlib.org/blas/. A CBLAS wrapper for Fortran BLAS
libraries is available from the same location.
