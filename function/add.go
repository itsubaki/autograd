package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Add(x ...*variable.Variable) []*variable.Variable {
	return (&Function{Forwarder: &AddT{}}).Apply(x...)
}

type AddT struct {
	x []variable.Data
}

func (f *AddT) Forward(x []variable.Data) []variable.Data {
	f.x = x

	add := func(a, b float64) float64 { return a + b }
	y := vector.F2(f.x[0], f.x[1], add)
	return []variable.Data{y}
}

func (f *AddT) Backward(gy []variable.Data) []variable.Data {
	return []variable.Data{gy[0], gy[0]}
}
