// cheb/cheb_test.go
package cheb

import (
	"math"
	"testing"
)

// tol is a tolerance for comparisons.
const tol = 1e-12

// tolAbs is an absolute tolerance for comparisons.
const tolAbs = 1e-11

// tolRel is a relative tolerance for comparisons.
const tolRel = 1e-9

// fSin is the function sin(x).
func fSin(x float64) float64 {
	return math.Sin(x)
}

// fT0 returns the constant 1.0.
func fT0(x float64) float64 {
	return 1.0
}

// fT1 returns x.
func fT1(x float64) float64 {
	return x
}

// fT2 returns 2*x^2 - 1.
func fT2(x float64) float64 {
	return 2*x*x - 1
}

// nearlyEqual compares two float64 values using both absolute and relative tolerances.
func nearlyEqual(a, b, absTol, relTol float64) bool {
	diff := math.Abs(a - b)
	if diff <= absTol {
		return true
	}
	// Use relative tolerance.
	absA := math.Abs(a)
	absB := math.Abs(b)
	largest := absA
	if absB > largest {
		largest = absB
	}
	return diff <= largest*relTol
}

// TestChebSeriesSin tests the Chebyshev series approximation for sin(x).
func TestChebSeriesSin(t *testing.T) {
	order := 40
	cs, err := NewChebSeries(order)
	if err != nil {
		t.Fatalf("NewChebSeries failed: %v", err)
	}
	if err := cs.ChebInit(fSin, -math.Pi, math.Pi); err != nil {
		t.Fatalf("ChebInit failed: %v", err)
	}

	// Test evaluation at several points in [-π, π].
	for x := -math.Pi; x <= math.Pi; x += math.Pi / 10 {
		approx := cs.Eval(x)
		trueVal := math.Sin(x)
		if math.Abs(approx-trueVal) > tol {
			t.Errorf("Eval(%f): got %g, want %g", x, approx, trueVal)
		}

		// Also test evaluation with error estimate.
		res, abserr, err := cs.EvalErr(x)
		if err != nil {
			t.Errorf("EvalErr(%f) returned error: %v", x, err)
		}
		if math.Abs(res-trueVal) > tol {
			t.Errorf("EvalErr(%f): got %g, want %g", x, res, trueVal)
		}
		// The estimated error should be non-negative.
		if abserr < 0 {
			t.Errorf("EvalErr(%f): negative error estimate %g", x, abserr)
		}
	}
}

// TestChebSeriesDerivative tests the derivative calculation.
func TestChebSeriesDerivative(t *testing.T) {
	order := 40
	cs, err := NewChebSeries(order)
	if err != nil {
		t.Fatalf("NewChebSeries failed: %v", err)
	}
	if err := cs.ChebInit(fSin, -math.Pi, math.Pi); err != nil {
		t.Fatalf("ChebInit failed: %v", err)
	}

	deriv, err := NewChebSeries(order)
	if err != nil {
		t.Fatalf("NewChebSeries (derivative) failed: %v", err)
	}
	if err := CalcDeriv(deriv, cs); err != nil {
		t.Fatalf("CalcDeriv failed: %v", err)
	}

	// The derivative of sin(x) is cos(x).
	for x := -math.Pi; x <= math.Pi; x += math.Pi / 10 {
		approx := deriv.Eval(x)
		trueVal := math.Cos(x)
		if !nearlyEqual(approx, trueVal, tolAbs, tolRel) {
			t.Errorf("Derivative Eval(%f): got %g, want %g", x, approx, trueVal)
		}
	}
}

// TestChebSeriesIntegral tests the integration calculation.
// The antiderivative of sin(x) with the condition of vanishing at x = -π is F(x) = -(1 + cos(x)).
func TestChebSeriesIntegral(t *testing.T) {
	order := 40
	cs, err := NewChebSeries(order)
	if err != nil {
		t.Fatalf("NewChebSeries failed: %v", err)
	}
	if err := cs.ChebInit(fSin, -math.Pi, math.Pi); err != nil {
		t.Fatalf("ChebInit failed: %v", err)
	}

	integ, err := NewChebSeries(order)
	if err != nil {
		t.Fatalf("NewChebSeries (integral) failed: %v", err)
	}
	if err := CalcInteg(integ, cs); err != nil {
		t.Fatalf("CalcInteg failed: %v", err)
	}

	// The antiderivative should be F(x) = -(1 + cos(x)) such that F(-π) = 0.
	for x := -math.Pi; x <= math.Pi; x += math.Pi / 10 {
		approx := integ.Eval(x)
		trueVal := -(1 + math.Cos(x))
		if math.Abs(approx-trueVal) > tol {
			t.Errorf("Integral Eval(%f): got %g, want %g", x, approx, trueVal)
		}
	}
}
