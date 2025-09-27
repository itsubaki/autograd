package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

type AdamW struct {
	Alpha       float64
	Beta1       float64
	Beta2       float64
	WeightDecay float64
	Hook        []Hook
	iter        int
	ms, vs      map[*variable.Variable]*tensor.Tensor[float64]
}

func (o *AdamW) Update(model Model) {
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

		update := tensor.F2(o.ms[p], o.vs[p], func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		decay := tensor.F(p.Data, func(w float64) float64 {
			return lr * o.WeightDecay * w
		})

		p.Data = tensor.Sub(p.Data, tensor.Add(update, decay))
	}
}
