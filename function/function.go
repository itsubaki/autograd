package function

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

type Forwarder interface {
	Forward(x []variable.Data) []variable.Data
	Backward(gy []variable.Data) []variable.Data
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
	data := make([]variable.Data, len(x))
	for i := range data {
		data[i] = x[i].Data
	}
	f.X, f.Y = data, f.Forward(data)

	f.in, f.out = x, make([]*variable.Variable, len(f.Y))
	for i := range f.out {
		f.out[i] = &variable.Variable{Data: f.Y[i], Creator: f}
	}

	return f.out
}

func (f Function) String() string {
	return fmt.Sprintf("%T(%v)", f.Forwarder, f.X)
}
