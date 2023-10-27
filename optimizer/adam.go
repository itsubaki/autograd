package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
)

type Adam struct {
	Alpha  float64
	Beta1  float64
	Beta2  float64
	iter   int
	ms, vs map[string]matrix.Matrix
	Hooks  []Hook
}

func (o *Adam) Update(model Model) {
	params := Params(model)
	for _, h := range o.Hooks {
		h(params)
	}

	if len(o.ms) == 0 {
		o.ms = make(map[string]matrix.Matrix)
		o.vs = make(map[string]matrix.Matrix)
	}

	o.iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	for _, p := range params {
		if _, ok := o.ms[id(p)]; !ok {
			o.ms[id(p)] = matrix.ZeroLike(p.Data)
			o.vs[id(p)] = matrix.ZeroLike(p.Data)
		}

		m, v := o.ms[id(p)], o.vs[id(p)]
		m = matrix.F2(m, p.Grad.Data, func(m, grad float64) float64 { return m + ((1 - o.Beta1) * (grad - m)) })      // m = m + (1-beta1 * (grad - m))
		v = matrix.F2(v, p.Grad.Data, func(v, grad float64) float64 { return v + ((1 - o.Beta2) * (grad*grad - v)) }) // v = v + (1-beta2 * (grad^2 - v))
		o.ms[id(p)], o.vs[id(p)] = m, v

		p.Data = matrix.Sub(p.Data, matrix.F2(m, v, func(a, b float64) float64 {
			return lr * a / (math.Sqrt(b) + 1e-8)
		}))
	}
}
