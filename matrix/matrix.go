package matrix

import (
	"math"
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/rand"
)

type Matrix [][]float64

func New(v ...[]float64) Matrix {
	out := make(Matrix, len(v))
	copy(out, v)
	return out
}

func Zero(m, n int) Matrix {
	out := make(Matrix, m)
	for i := range m {
		out[i] = make([]float64, n)
	}

	return out
}

func ZeroLike(m Matrix) Matrix {
	return Zero(Dim(m))
}

func OneLike(m Matrix) Matrix {
	return AddC(1.0, ZeroLike(m))
}

func From(x [][]int) Matrix {
	out := Zero(len(x), len(x[0]))
	for i := range x {
		for j := range x[i] {
			out[i][j] = float64(x[i][j])
		}
	}

	return out
}

// rnd returns a pseudo-random number generator.
func rnd(s ...randv2.Source) *randv2.Rand {
	if len(s) == 0 || s[0] == nil {
		return randv2.New(rand.NewSource(rand.MustRead()))
	}

	return randv2.New(s[0])
}

// Rand returns a matrix with elements that pseudo-random number in the half-open interval [0.0,1.0).
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Rand(m, n int, s ...randv2.Source) Matrix {
	return F(Zero(m, n), func(_ float64) float64 { return rnd(s...).Float64() })
}

// Randn returns a matrix with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Randn(m, n int, s ...randv2.Source) Matrix {
	return F(Zero(m, n), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

func Size(m Matrix) int {
	s := 1
	for _, v := range Shape(m) {
		s = s * v
	}

	return s
}

func Shape(m Matrix) []int {
	a, b := Dim(m)
	return []int{a, b}
}

func Dim(m Matrix) (int, int) {
	return len(m), len(m[0])
}

func AddC(c float64, m Matrix) Matrix {
	return F(m, func(v float64) float64 { return c + v })
}

// SubC returns c - m
func SubC(c float64, m Matrix) Matrix {
	return F(m, func(v float64) float64 { return c - v })
}

func MulC(c float64, m Matrix) Matrix {
	return F(m, func(v float64) float64 { return c * v })
}

func Exp(m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Exp(v) })
}

func Log(m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Log(v) })
}

func Sin(m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Sin(v) })
}

func Cos(m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Cos(v) })
}

func Tanh(m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Tanh(v) })
}

func Pow(c float64, m Matrix) Matrix {
	return F(m, func(v float64) float64 { return math.Pow(v, c) })
}

func Add(m, n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a + b })
}

func Sub(m, n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a - b })
}

func Mul(m, n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a * b })
}

func Div(m, n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a / b })
}

func Mean(m Matrix) float64 {
	return Sum(m) / float64(Size(m))
}

func Sum(m Matrix) float64 {
	var sum float64
	for _, v := range Flatten(m) {
		sum = sum + v
	}

	return sum
}

func Max(m Matrix) float64 {
	max := m[0][0]
	for _, v := range Flatten(m) {
		if v > max {
			max = v
		}
	}

	return max
}

func Min(m Matrix) float64 {
	min := m[0][0]
	for _, v := range Flatten(m) {
		if v < min {
			min = v
		}
	}

	return min
}

func Argmax(m Matrix) []int {
	p, q := Dim(m)

	out := make([]int, p)
	for i := range p {
		max := m[i][0]
		for j := range q {
			if m[i][j] > max {
				max, out[i] = m[i][j], j
			}
		}
	}

	return out
}

// Dot returns the dot product of m and n.
func Dot(m, n Matrix) Matrix {
	a, b := Dim(m)
	_, p := Dim(n)

	out := Zero(a, p)
	for i := range a {
		for j := range p {
			for k := 0; k < b; k++ {
				out[i][j] = out[i][j] + m[i][k]*n[k][j]
			}
		}
	}

	return out
}

func Clip(m Matrix, min, max float64) Matrix {
	return F(m, func(v float64) float64 {
		if v < min {
			return min
		}

		if v > max {
			return max
		}

		return v
	})
}

// Mask returns a matrix with elements that 1 if f() is true and 0 otherwise.
func Mask(m Matrix, f func(x float64) bool) Matrix {
	return F(m, func(v float64) float64 {
		if f(v) {
			return 1
		}

		return 0
	})
}

func Broadcast(m, n Matrix) (Matrix, Matrix) {
	return BroadcastTo(Shape(n), m), BroadcastTo(Shape(m), n)
}

func BroadcastTo(shape []int, m Matrix) Matrix {
	a, b := shape[0], shape[1]

	if len(m) == 1 && len(m[0]) == 1 {
		out := make([]float64, a*b)
		for i := range a * b {
			out[i] = m[0][0]
		}

		return Reshape(shape, New(out))
	}

	if len(m) == 1 {
		// b is ignored
		out := make(Matrix, a)
		for i := range a {
			out[i] = m[0]
		}

		return out
	}

	if len(m[0]) == 1 {
		// a is ignored
		out := Zero(len(m), b)
		for i := range m {
			for j := range b {
				out[i][j] = m[i][0]
			}
		}

		return out
	}

	return m
}

func SumTo(shape []int, m Matrix) Matrix {
	if shape[0] == 1 && shape[1] == 1 {
		return New([]float64{Sum(m)})
	}

	if shape[0] == 1 {
		return SumAxis0(m)
	}

	if shape[1] == 1 {
		return SumAxis1(m)
	}

	return m
}

// SumAxis0 returns the sum of each column.
func SumAxis0(m Matrix) Matrix {
	p, q := Dim(m)

	v := make([]float64, 0, q)
	for j := range q {
		var sum float64
		for i := range p {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return New(v)
}

// SumAxis1 returns the sum of each row.
func SumAxis1(m Matrix) Matrix {
	p, q := Dim(m)

	v := make([]float64, 0, p)
	for i := range p {
		var sum float64
		for j := range q {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return Transpose(New(v))
}

func MaxAxis1(m Matrix) Matrix {
	p, q := Dim(m)

	v := make([]float64, 0, p)
	for i := range p {
		max := m[i][0]
		for j := range q {
			if m[i][j] > max {
				max = m[i][j]
			}
		}

		v = append(v, max)
	}

	return Transpose(New(v))
}

func Transpose(m Matrix) Matrix {
	p, q := Dim(m)

	out := Zero(q, p)
	for i := range q {
		for j := range p {
			out[i][j] = m[j][i]
		}
	}

	return out
}

// Reshape returns the matrix with the given shape.
func Reshape(shape []int, m Matrix) Matrix {
	p, q := Dim(m)
	a, b := shape[0], shape[1]

	v := Flatten(m)
	if a < 1 {
		a = p * q / b
	}

	if b < 1 {
		b = p * q / a
	}

	out := make(Matrix, a)
	for i := range a {
		out[i] = v[i*b : (i+1)*b]
	}

	return out
}

func Flatten(m Matrix) []float64 {
	out := make([]float64, 0, Size(m))
	for _, r := range m {
		out = append(out, r...)
	}

	return out
}

func F(m Matrix, f func(a float64) float64) Matrix {
	p, q := Dim(m)

	out := Zero(p, q)
	for i := range p {
		for j := range q {
			out[i][j] = f(m[i][j])
		}
	}

	return out
}

func F2(m, n Matrix, f func(a, b float64) float64) Matrix {
	x, y := Broadcast(m, n)
	p, q := Dim(x)

	out := Zero(p, q)
	for i := range p {
		for j := range q {
			out[i][j] = f(x[i][j], y[i][j])
		}
	}

	return out
}
