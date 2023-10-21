package optimizer

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type Adam struct {
	Alpha  float64
	Beta1  float64
	Beta2  float64
	iter   int
	ms, vs map[string]matrix.Matrix
	Hooks  []Hook
}

func (o *Adam) LearningRate() float64 {
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	return o.Alpha * math.Sqrt(fix2) / fix1
}

func (o *Adam) Update(model Model) {
	o.iter++
	if len(o.ms) == 0 {
		o.ms = make(map[string]matrix.Matrix)
		o.vs = make(map[string]matrix.Matrix)
	}

	params := Params(model)
	for _, h := range o.Hooks {
		h(params)
	}

	for _, p := range params {
		if _, ok := o.ms[id(p)]; !ok {
			o.ms[id(p)] = matrix.ZeroLike(p.Data)
			o.vs[id(p)] = matrix.ZeroLike(p.Data)
		}

		m, v := o.ms[id(p)], o.vs[id(p)]
		m = matrix.Add(m, matrix.MulC(1-o.Beta1, matrix.Sub(p.Grad.Data, m)))                // m = m + (1-beta1 * (grad - m))
		v = matrix.Add(v, matrix.MulC(1-o.Beta2, matrix.Sub(matrix.Pow(2, p.Grad.Data), v))) // v = v + (1-beta2 * (grad^2 - v))

		p.Data = matrix.Sub(p.Data, matrix.F2(m, v, func(a, b float64) float64 {
			return o.LearningRate() * a / (math.Sqrt(b) + 1e-8)
		}))
	}
}

func id(p *variable.Variable) string {
	return fmt.Sprintf("%p", p)
}
