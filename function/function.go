package function

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

type Forwarder interface {
	Forward(x ...variable.Data) []variable.Data
	Backward(gy ...variable.Data) []variable.Data
}

type Function struct {
	X, Y    []variable.Data
	in, out []*variable.Variable
	Forwarder
}

func (f *Function) Input() []*variable.Variable {
	return f.in
}

func (f *Function) Output() []*variable.Variable {
	return f.out
}

func (f *Function) Apply(x ...*variable.Variable) []*variable.Variable {
	data := xdata(x)

	f.X, f.Y = data, f.Forward(data...)
	f.in, f.out = x, yvariable(f.Y, f)
	return f.out
}

func (f Function) String() string {
	return fmt.Sprintf("%T(%v)", f.Forwarder, f.X)
}

func xdata(x []*variable.Variable) []variable.Data {
	data := make([]variable.Data, len(x))
	for i := range x {
		data[i] = x[i].Data
	}

	return data
}

func yvariable(y []variable.Data, f *Function) []*variable.Variable {
	yvar := make([]*variable.Variable, len(y))
	for i := range y {
		yvar[i] = &variable.Variable{Data: y[i], Creator: f}
	}

	return yvar
}
