package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// AdamW is an Adam optimizer with decoupled weight decay.
type AdamW struct {
	Alpha       float64
	Beta1       float64
	Beta2       float64
	WeightDecay float64
	Hook        []Hook
	iter        int
	ms, vs      map[*variable.Variable]*tensor.Tensor[float64]
}

// Update updates the parameters of the model.
func (o *AdamW) Update(model Model) {
	if len(o.ms) == 0 {
		o.ms = make(map[*variable.Variable]*tensor.Tensor[float64])
		o.vs = make(map[*variable.Variable]*tensor.Tensor[float64])
	}

	o.iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	params := Params(model, o.Hook)
	for _, p := range params {
		if _, ok := o.ms[p]; !ok {
			o.ms[p] = tensor.ZeroLike(p.Data)
			o.vs[p] = tensor.ZeroLike(p.Data)
		}

		o.ms[p] = tensor.F2(o.ms[p], p.Grad.Data, func(m, g float64) float64 {
			return m + (1-o.Beta1)*(g-m)
		})

		o.vs[p] = tensor.F2(o.vs[p], p.Grad.Data, func(v, g float64) float64 {
			return v + (1-o.Beta2)*(g*g-v)
		})

		step := tensor.F2(o.ms[p], o.vs[p], func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		p.Data = tensor.Sub(p.Data, step)
		p.Data = tensor.F(p.Data, func(w float64) float64 {
			return w * (1 - lr*o.WeightDecay)
		})
	}
}
