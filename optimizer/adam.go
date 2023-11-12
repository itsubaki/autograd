package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type Adam struct {
	Alpha  float64
	Beta1  float64
	Beta2  float64
	Hook   []Hook
	iter   int
	ms, vs map[*variable.Variable]matrix.Matrix
}

func (o *Adam) Update(model Model) {
	params := Params(model, o.Hook)

	if len(o.ms) == 0 {
		o.ms = make(map[*variable.Variable]matrix.Matrix)
		o.vs = make(map[*variable.Variable]matrix.Matrix)
	}

	o.iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	for _, p := range params {
		if _, ok := o.ms[p]; !ok {
			o.ms[p] = matrix.ZeroLike(p.Data)
			o.vs[p] = matrix.ZeroLike(p.Data)
		}

		o.ms[p] = matrix.F2(o.ms[p], p.Grad.Data, func(m, grad float64) float64 { return m + ((1 - o.Beta1) * (grad - m)) })      // m = m + ((1-beta1) * (grad - m))
		o.vs[p] = matrix.F2(o.vs[p], p.Grad.Data, func(v, grad float64) float64 { return v + ((1 - o.Beta2) * (grad*grad - v)) }) // v = v + ((1-beta2) * (grad^2 - v))

		// param = param - (lr * m / (sqrt(v) + 1e-8))
		p.Data = matrix.Sub(p.Data, matrix.F2(o.ms[p], o.vs[p], func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		}))
	}
}
