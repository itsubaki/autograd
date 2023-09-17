package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sub(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SubT{}}).Apply(x...)[0]
}

type SubT struct{}

func (f *SubT) Forward(x ...variable.Data) []variable.Data {
	y := vector.Sub(x[0], x[1])
	return []variable.Data{y}
}

func (f *SubT) Backward(gy ...variable.Data) []variable.Data {
	return []variable.Data{gy[0], vector.MulC(gy[0], -1.0)}
}
