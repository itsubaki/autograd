package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sum(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SumT{}}).ApplyAndFirst(x...)
}

type SumT struct {
	x *variable.Variable
}

func (f *SumT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := vector.Sum(f.x.Data)
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *SumT) Backward(gy ...*variable.Variable) []*variable.Variable {
	y, _ := vector.Broadcast(gy[0].Data, f.x.Data)
	return []*variable.Variable{
		variable.New(y...),
	}
}
