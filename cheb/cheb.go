// cheb/cheb.go
//
// Package cheb provides functions to work with Chebyshev series.
// It offers evaluation routines, as well as routines to calculate the derivative
// and integral of a Chebyshev series.
//
// This code is a transpilation of the C code originally written by Gerard Jungman and others.
// The Go code uses standard Go error handling and idioms.
package cheb

import (
	"errors"
	"fmt"
	"math"
)

// Machine epsilon for float64.
const dblEpsilon = 2.220446049250313e-16

// Mode represents the precision mode used when evaluating a Chebyshev series.
type Mode int

const (
	// PrecDouble uses full (double) precision.
	PrecDouble Mode = iota
	// PrecSingle uses single precision (if available).
	PrecSingle
)

// Function defines the type for functions to be approximated.
type Function func(x float64) float64

// ChebSeries holds the Chebyshev series coefficients and parameters.
type ChebSeries struct {
	C       []float64 // Coefficients (length = order+1)
	Order   int       // Order of the expansion
	A, B    float64   // Interval [A, B]
	OrderSp int       // Effective order for single precision evaluation
	F       []float64 // Function values at Chebyshev points (length = order+1)
}

// NewChebSeries allocates and returns a new ChebSeries for the given order.
func NewChebSeries(order int) (*ChebSeries, error) {
	if order < 0 {
		return nil, errors.New("order must be nonnegative")
	}
	cs := &ChebSeries{
		Order:   order,
		OrderSp: order,
		C:       make([]float64, order+1),
		F:       make([]float64, order+1),
	}
	return cs, nil
}

// Eval evaluates the Chebyshev series at x.
// It uses the full order stored in the ChebSeries.
func (cs *ChebSeries) Eval(x float64) float64 {
	y := (2.0*x - cs.A - cs.B) / (cs.B - cs.A)
	y2 := 2.0 * y
	var d1, d2 float64
	// Loop from cs.Order downto 1.
	for i := cs.Order; i >= 1; i-- {
		temp := d1
		d1 = y2*d1 - d2 + cs.C[i]
		d2 = temp
	}
	return y*d1 - d2 + 0.5*cs.C[0]
}

// EvalN evaluates the Chebyshev series at x using at most n coefficients.
func (cs *ChebSeries) EvalN(n int, x float64) float64 {
	evalOrder := min(n, cs.Order)
	y := (2.0*x - cs.A - cs.B) / (cs.B - cs.A)
	y2 := 2.0 * y
	var d1, d2 float64
	for i := evalOrder; i >= 1; i-- {
		temp := d1
		d1 = y2*d1 - d2 + cs.C[i]
		d2 = temp
	}
	return y*d1 - d2 + 0.5*cs.C[0]
}

// EvalErr evaluates the Chebyshev series at x and estimates the error.
// It returns the result, the estimated absolute error, and an error if one occurred.
func (cs *ChebSeries) EvalErr(x float64) (result float64, abserr float64, err error) {
	y := (2.0*x - cs.A - cs.B) / (cs.B - cs.A)
	y2 := 2.0 * y
	var d1, d2 float64
	for i := cs.Order; i >= 1; i-- {
		temp := d1
		d1 = y2*d1 - d2 + cs.C[i]
		d2 = temp
	}
	result = y*d1 - d2 + 0.5*cs.C[0]

	// Estimate cumulative numerical error.
	var absc float64
	for i := 0; i <= cs.Order; i++ {
		absc += math.Abs(cs.C[i])
	}
	abserr = math.Abs(cs.C[cs.Order]) + absc*dblEpsilon

	return result, abserr, nil
}

// EvalNErr evaluates the Chebyshev series at x using at most n coefficients and estimates the error.
func (cs *ChebSeries) EvalNErr(n int, x float64) (result float64, abserr float64, err error) {
	evalOrder := min(n, cs.Order)
	y := (2.0*x - cs.A - cs.B) / (cs.B - cs.A)
	y2 := 2.0 * y
	var d1, d2 float64
	for i := evalOrder; i >= 1; i-- {
		temp := d1
		d1 = y2*d1 - d2 + cs.C[i]
		d2 = temp
	}
	result = y*d1 - d2 + 0.5*cs.C[0]

	// Estimate cumulative numerical error.
	var absc float64
	for i := 0; i <= evalOrder; i++ {
		absc += math.Abs(cs.C[i])
	}
	abserr = math.Abs(cs.C[evalOrder]) + absc*dblEpsilon

	return result, abserr, nil
}

// EvalModeE evaluates the Chebyshev series at x with a specified mode (precision)
// and estimates the error.
func (cs *ChebSeries) EvalModeE(x float64, mode Mode) (result float64, abserr float64, err error) {
	var evalOrder int
	if mode == PrecDouble {
		evalOrder = cs.Order
	} else {
		evalOrder = cs.OrderSp
	}
	y := (2.0*x - cs.A - cs.B) / (cs.B - cs.A)
	y2 := 2.0 * y
	var d1, d2 float64
	for i := evalOrder; i >= 1; i-- {
		temp := d1
		d1 = y2*d1 - d2 + cs.C[i]
		d2 = temp
	}
	result = y*d1 - d2 + 0.5*cs.C[0]

	// Estimate cumulative numerical error.
	var absc float64
	for i := 0; i <= evalOrder; i++ {
		absc += math.Abs(cs.C[i])
	}
	abserr = math.Abs(cs.C[evalOrder]) + absc*dblEpsilon

	return result, abserr, nil
}

// EvalMode evaluates the Chebyshev series at x with a specified mode.
// It panics if an error is encountered (mirroring the behavior of the C version).
func (cs *ChebSeries) EvalMode(x float64, mode Mode) float64 {
	res, _, err := cs.EvalModeE(x, mode)
	if err != nil {
		panic(fmt.Sprintf("cheb: EvalMode: %v", err))
	}
	return res
}

// ChebInit initializes the Chebyshev series cs to approximate the given function f on [a, b].
// It computes the coefficients using the standard Chebyshev cosine transforms.
func (cs *ChebSeries) ChebInit(f Function, a, b float64) error {
	if a >= b {
		return errors.New("cheb: invalid interval: a must be less than b")
	}
	cs.A = a
	cs.B = b
	bma := 0.5 * (cs.B - cs.A)
	bpa := 0.5 * (cs.B + cs.A)
	// fac = 2/(order+1)
	fac := 2.0 / float64(cs.Order+1)

	// Evaluate the function at the Chebyshev points.
	for k := 0; k <= cs.Order; k++ {
		angle := math.Pi * (float64(k) + 0.5) / float64(cs.Order+1)
		y := math.Cos(angle)
		cs.F[k] = f(y*bma + bpa)
	}

	// Compute coefficients.
	for j := 0; j <= cs.Order; j++ {
		var sum float64
		for k := 0; k <= cs.Order; k++ {
			angle := math.Pi * float64(j) * (float64(k) + 0.5) / float64(cs.Order+1)
			sum += cs.F[k] * math.Cos(angle)
		}
		cs.C[j] = fac * sum
	}

	return nil
}

// Order returns the order of the Chebyshev series.
func (cs *ChebSeries) OrderValue() int {
	return cs.Order
}

// Size returns the size of the coefficient array (order + 1).
func (cs *ChebSeries) Size() int {
	return cs.Order + 1
}

// Coeffs returns the coefficients slice.
func (cs *ChebSeries) Coeffs() []float64 {
	return cs.C
}

// CalcInteg calculates the integral (antiderivative) of the Chebyshev series f
// and stores the result in integ. The integral is fixed so that it vanishes at the left end-point.
func CalcInteg(integ, f *ChebSeries) error {
	n := f.Order + 1
	con := 0.25 * (f.B - f.A)

	// Check orders match.
	if integ.Order != f.Order {
		return errors.New("cheb: orders of chebyshev series must be equal")
	}
	integ.A = f.A
	integ.B = f.B

	if n == 1 {
		integ.C[0] = 0.0
	} else if n == 2 {
		integ.C[1] = con * f.C[0]
		integ.C[0] = 2.0 * integ.C[1]
	} else {
		var sum float64
		fac := 1.0
		for i := 1; i <= n-2; i++ {
			integ.C[i] = con * (f.C[i-1] - f.C[i+1]) / float64(i)
			sum += fac * integ.C[i]
			fac = -fac
		}
		integ.C[n-1] = con * f.C[n-2] / float64(n-1)
		sum += fac * integ.C[n-1]
		integ.C[0] = 2.0 * sum
	}

	return nil
}

// CalcDeriv calculates the derivative of the Chebyshev series f
// and stores the result in deriv.
func CalcDeriv(deriv, f *ChebSeries) error {
	n := f.Order + 1
	con := 2.0 / (f.B - f.A)

	if deriv.Order != f.Order {
		return errors.New("cheb: orders of chebyshev series must be equal")
	}
	deriv.A = f.A
	deriv.B = f.B

	// The last coefficient is set to 0.
	deriv.C[n-1] = 0.0

	if n > 1 {
		deriv.C[n-2] = 2.0 * float64(n-1) * f.C[n-1]
		// Loop i = n downto 3 (i.e. index i-3 from n-3 downto 0).
		for i := n; i >= 3; i-- {
			deriv.C[i-3] = deriv.C[i-1] + 2.0*float64(i-2)*f.C[i-2]
		}
		// Multiply all coefficients by con.
		for i := 0; i < n; i++ {
			deriv.C[i] *= con
		}
	}
	return nil

}
