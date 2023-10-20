package layer

import "github.com/itsubaki/autograd/variable"

var (
	_ Layer = (*LinearT)(nil)
	_ Layer = (*RNNT)(nil)
	_ Layer = (*LSTMT)(nil)
)

type Layer interface {
	First(x ...*variable.Variable) *variable.Variable
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() []Parameter
	Cleargrads()
}

type Layers map[string]Layer

func (l Layers) Add(name string, layer Layer) {
	l[name] = layer
}

func (l Layers) Params() []Parameter {
	params := make([]Parameter, 0)
	for k := range l {
		params = append(params, l[k].Params()...)
	}

	return params
}

func (l Layers) FlattenParams() Parameters {
	out := make(Parameters)
	for k := range l {
		for _, p := range l[k].Params() {
			out[k+"."+p.Name] = p
		}
	}

	return out
}

func (l Layers) Cleargrads() {
	for k := range l {
		l[k].Cleargrads()
	}
}
