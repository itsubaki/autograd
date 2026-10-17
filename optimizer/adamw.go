package optimizer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/math"
	"github.com/itsubaki/autograd/tensor"
)

// AdamW is an Adam optimizer with decoupled weight decay.
type AdamW struct {
	Alpha       float32
	Beta1       float32
	Beta2       float32
	WeightDecay float32
	Iter        int
	Ms          map[string]*tensor.Tensor[float32]
	Vs          map[string]*tensor.Tensor[float32]
}

// Update updates the parameters of the model.
func (o *AdamW) Update(params layer.Parameters) {
	if len(o.Ms) == 0 {
		o.Ms = make(map[string]*tensor.Tensor[float32])
		o.Vs = make(map[string]*tensor.Tensor[float32])
	}

	o.Iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float32(o.Iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float32(o.Iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	for name, p := range params {
		if p.Grad == nil {
			continue
		}

		if _, ok := o.Ms[name]; !ok {
			o.Ms[name] = tensor.ZerosLike(p.Data)
			o.Vs[name] = tensor.ZerosLike(p.Data)
		}

		o.Ms[name] = tensor.F2(o.Ms[name], p.Grad.Data, func(m, g float32) float32 {
			return m + (1-o.Beta1)*(g-m)
		})

		o.Vs[name] = tensor.F2(o.Vs[name], p.Grad.Data, func(v, g float32) float32 {
			return v + (1-o.Beta2)*(g*g-v)
		})

		step := tensor.F2(o.Ms[name], o.Vs[name], func(m, v float32) float32 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		p.Data = tensor.Sub(p.Data, step)
		p.Data = tensor.F(p.Data, func(w float32) float32 {
			return w * (1 - lr*o.WeightDecay)
		})
	}
}
