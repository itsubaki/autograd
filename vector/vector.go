package vector

import "math"

func ZeroLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return AddC(1.0, ZeroLike(v))
}

func Const(c float64) []float64 {
	return []float64{c}
}

func Shape(v []float64) []int {
	return []int{1, len(v)}
}

func AddC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c + a })
}

// SubC returns c - v
func SubC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c - a })
}

func MulC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c * a })
}

func Exp(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

func Log(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Log(a) })
}

func Sin(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

func Cos(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

func Tanh(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Tanh(a) })
}

func Pow(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Pow(a, c) })
}

func Add(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a + b })
}

// Sub returns v - w
func Sub(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a - b })
}

func Mul(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a * b })
}

// Div returns v / w
func Div(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a / b })
}

func BroadcastTo(shape []int, v []float64) []float64 {
	// NOTE: v is vector, shape[0] has no effect. shape[0] is always 1.
	out, _ := Broadcast(v, make([]float64, shape[1]))
	return out
}

func Broadcast(v, w []float64) ([]float64, []float64) {
	if len(v) == 1 {
		out := ZeroLike(w)
		for i := 0; i < len(w); i++ {
			out[i] = v[0]
		}

		return out, w
	}

	if len(w) == 1 {
		out := ZeroLike(v)
		for i := 0; i < len(v); i++ {
			out[i] = w[0]
		}

		return v, out
	}

	return v, w
}

func SumTo(shape []int, v []float64) float64 {
	// NOTE: v is vector, shape has no effect.
	return Sum(v)
}

func Sum(v []float64) float64 {
	var sum float64
	for i := range v {
		sum += v[i]
	}

	return sum
}

func F(v []float64, f func(a float64) float64) []float64 {
	out := ZeroLike(v)
	for i := range v {
		out[i] = f(v[i])
	}

	return out
}

func F2(v, w []float64, f func(a, b float64) float64) []float64 {
	out := ZeroLike(v)
	for i := range v {
		out[i] = f(v[i], w[i])
	}

	return out
}

func Equals[T comparable](v, w []T) bool {
	if len(v) != len(w) {
		return false
	}

	for i := range v {
		if v[i] != w[i] {
			return false
		}
	}

	return true
}
