package vector

import "math"

func NewLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return ConstLike(1.0, v)
}

func ConstLike(c float64, v []float64) []float64 {
	return AddC(NewLike(v), c)
}

func AddC(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return a + c })
}

func SubC(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return a - c })
}

func MulC(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return c * a })
}

func Exp(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

func Sin(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

func Cos(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

func Pow(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return math.Pow(a, c) })
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
	out := NewLike(v)
	for i := range v {
		out[i] = f(v[i], w[i])
	}

	return out
}
