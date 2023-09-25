package vector

import "math"

func NewLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return AddC(1.0, NewLike(v))
}

func Const(c float64) []float64 {
	return []float64{c}
}

func Broadcast(v, w []float64) ([]float64, []float64) {
	if len(v) == 1 {
		out := NewLike(w)
		for i := 0; i < len(w); i++ {
			out[i] = v[0]
		}

		return out, w
	}

	if len(w) == 1 {
		out := NewLike(v)
		for i := 0; i < len(v); i++ {
			out[i] = w[0]
		}

		return v, out
	}

	return v, w
}

func AddC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c + a })
}

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

func Sum(v []float64) float64 {
	var sum float64
	for i := range v {
		sum += v[i]
	}

	return sum
}

func Add(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a + b })
}

func Sub(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a - b })
}

func Mul(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a * b })
}

func Div(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a / b })
}

func F(v []float64, f func(a float64) float64) []float64 {
	out := NewLike(v)
	for i := range v {
		out[i] = f(v[i])
	}

	return out
}

func F2(v, w []float64, f func(a, b float64) float64) []float64 {
	v, w = Broadcast(v, w)

	out := NewLike(v)
	for i := range v {
		out[i] = f(v[i], w[i])
	}

	return out
}
