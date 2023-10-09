package matrix

import (
	"math"
	"math/rand"
	"time"
)

type Matrix [][]float64

func New(v ...[]float64) Matrix {
	out := make(Matrix, len(v))
	copy(out, v)
	return out
}

func Zero(m, n int) Matrix {
	out := make(Matrix, m)
	for i := 0; i < m; i++ {
		out[i] = make([]float64, n)
	}

	return out
}

func ZeroLike(m Matrix) Matrix {
	s := Shape(m)
	return Zero(s[0], s[1])
}

func OneLike(m Matrix) Matrix {
	return AddC(1.0, ZeroLike(m))
}

func Const(c float64) Matrix {
	return [][]float64{{c}}
}

// Rand returns a matrix with elements that pseudo-random number in the half-open interval [0.0,1.0).
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Rand(m, n int, s ...rand.Source) Matrix {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	return F(Zero(m, n), func(_ float64) float64 { return rng.Float64() })
}

// Randn returns a matrix with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Randn(m, n int, s ...rand.Source) Matrix {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	return F(Zero(m, n), func(_ float64) float64 { return rng.NormFloat64() })
}

func Shape(m Matrix) []int {
	return []int{len(m), len(m[0])}
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

func MaxAxis1(m Matrix) Matrix {
	s := Shape(m)
	p, q := s[0], s[1]

	v := make([]float64, 0, p)
	for i := 0; i < p; i++ {
		var max float64
		for j := 0; j < q; j++ {
			if m[i][j] > max {
				max = m[i][j]
			}
		}

		v = append(v, max)
	}

	return Transpose(New(v))
}

func Max(m Matrix) Matrix {
	s := Shape(m)
	p, q := s[0], s[1]

	max := m[0][0]
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			if m[i][j] > max {
				max = m[i][j]
			}
		}
	}

	return New([]float64{max})
}

func Min(m Matrix) Matrix {
	s := Shape(m)
	p, q := s[0], s[1]

	min := m[0][0]
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			if m[i][j] < min {
				min = m[i][j]
			}
		}
	}

	return New([]float64{min})
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

// Dot returns the dot product of m and n.
func Dot(m, n Matrix) Matrix {
	msh, nsh := Shape(m), Shape(n)
	a, b, p := msh[0], msh[1], nsh[1]

	out := Zero(a, p)
	for i := 0; i < a; i++ {
		for j := 0; j < p; j++ {
			for k := 0; k < b; k++ {
				out[i][j] = out[i][j] + m[i][k]*n[k][j]
			}
		}
	}

	return out
}

func Broadcast(m, n Matrix) (Matrix, Matrix) {
	return BroadcastTo(Shape(n), m), BroadcastTo(Shape(m), n)
}

func BroadcastTo(shape []int, m Matrix) Matrix {
	a, b := shape[0], shape[1]
	if len(m) == 1 && len(m[0]) == 1 {
		out := Zero(a, b)
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				out[i][j] = m[0][0]
			}
		}

		return out
	}

	if len(m) == 1 {
		// b is ignored
		out := make(Matrix, a)
		for i := 0; i < a; i++ {
			out[i] = m[0]
		}

		return out
	}

	if len(m[0]) == 1 {
		// a is ignored
		out := Zero(len(m), b)
		for i := 0; i < len(m); i++ {
			for j := 0; j < b; j++ {
				out[i][j] = m[i][0]
			}
		}

		return out
	}

	return m
}

func Sum(m Matrix) float64 {
	var sum float64
	for i := range m {
		for j := range m[i] {
			sum = sum + m[i][j]
		}
	}

	return sum
}

func SumTo(shape []int, m Matrix) Matrix {
	if shape[0] == 1 && shape[1] == 1 {
		return Const(Sum(m))
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
	s := Shape(m)
	p, q := s[0], s[1]

	v := make([]float64, 0, q)
	for j := 0; j < q; j++ {
		var sum float64
		for i := 0; i < p; i++ {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return New(v)
}

// SumAxis1 returns the sum of each row.
func SumAxis1(m Matrix) Matrix {
	s := Shape(m)
	p, q := s[0], s[1]

	v := make([]float64, 0, p)
	for i := 0; i < p; i++ {
		var sum float64
		for j := 0; j < q; j++ {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return Transpose(New(v))
}

func Transpose(m Matrix) Matrix {
	shape := Shape(m)
	p, q := shape[0], shape[1]

	out := Zero(q, p)
	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			out[i][j] = m[j][i]
		}
	}

	return out
}

// Reshape returns the matrix with the given shape.
func Reshape(shape []int, x Matrix) Matrix {
	xsh := Shape(x)
	p, q, a, b := xsh[0], xsh[1], shape[0], shape[1]

	v := Flatten(x)
	if a < 1 {
		a = p * q / b
	}

	if b < 1 {
		b = p * q / a
	}

	out := New()
	for i := 0; i < a; i++ {
		begin, end := i*b, (i+1)*b
		out = append(out, v[begin:end])
	}

	return out
}

func Flatten(m Matrix) []float64 {
	out := make([]float64, 0)
	for _, r := range m {
		out = append(out, r...)
	}

	return out
}

func F(m Matrix, f func(a float64) float64) Matrix {
	shape := Shape(m)
	p, q := shape[0], shape[1]

	out := Zero(p, q)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			out[i][j] = f(m[i][j])
		}
	}

	return out
}

func F2(m, n Matrix, f func(a, b float64) float64) Matrix {
	shape := Shape(m)
	p, q := shape[0], shape[1]

	out := Zero(p, q)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			out[i][j] = f(m[i][j], n[i][j])
		}
	}

	return out
}

func Equals(m, n Matrix) bool {
	s0, s1 := Shape(m), Shape(n)
	p, q, a, b := s0[0], s0[1], s1[0], s1[1]

	if p != a || q != b {
		return false
	}

	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			if m[i][j] != n[i][j] {
				return false
			}
		}
	}

	return true
}
