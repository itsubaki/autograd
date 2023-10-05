package matrix

import "math"

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

	return T(New(v))
}

func T(m Matrix) Matrix {
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
