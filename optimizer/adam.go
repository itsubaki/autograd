package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

type Adam struct {
	Alpha  float64
	Beta1  float64
	Beta2  float64
	Hook   []Hook
	iter   int
	ms, vs map[*variable.Variable]*tensor.Tensor[float64]
}

func (o *Adam) Update(model Model) {
	o.update(model, func(lr float64, data, ms, vs *tensor.Tensor[float64]) *tensor.Tensor[float64] {
		return tensor.F2(ms, vs, func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})
	})
}

func (o *Adam) update(model Model, f func(lr float64, data, ms, vs *tensor.Tensor[float64]) *tensor.Tensor[float64]) {
	params := Params(model, o.Hook)

	if len(o.ms) == 0 {
		o.ms = make(map[*variable.Variable]*tensor.Tensor[float64])
		o.vs = make(map[*variable.Variable]*tensor.Tensor[float64])
	}

	o.iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	for _, p := range params {
		if _, ok := o.ms[p]; !ok {
			o.ms[p] = tensor.ZeroLike(p.Data)
			o.vs[p] = tensor.ZeroLike(p.Data)
		}

		o.ms[p] = tensor.F2(o.ms[p], p.Grad.Data, func(m, grad float64) float64 {
			return m + (1-o.Beta1)*(grad-m)
		})

		o.vs[p] = tensor.F2(o.vs[p], p.Grad.Data, func(v, grad float64) float64 {
			return v + (1-o.Beta2)*(grad*grad-v)
		})

		// update function
		update := f(lr, p.Data, o.ms[p], o.vs[p])

		// param = param - (lr * m / (sqrt(v) + 1e-8))
		p.Data = tensor.Sub(p.Data, update)
	}
}
