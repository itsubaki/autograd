package vector

func NewLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return AddC(NewLike(v), 1)
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

func Add(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a + b })
}

func Sub(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a - b })
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
