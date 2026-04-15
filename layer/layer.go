package layer

import "github.com/itsubaki/autograd/variable"

// Layer is the interface implemented by trainable layers.
type Layer interface {
	First(x ...*variable.Variable) *variable.Variable
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() Parameters
	Cleargrads()
}

// Layers is a named collection of layers.
type Layers map[string]Layer

// Add stores a layer under the given name.
func (l Layers) Add(name string, layer Layer) {
	l[name] = layer
}

// Params returns the parameters of all layers in the collection.
func (l Layers) Params() Parameters {
	params := make(Parameters)
	for k := range l {
		for _, p := range l[k].Params() {
			params[k+"."+p.Name] = p
		}
	}

	return params
}

// Cleargrads clears gradients for all layers in the collection.
func (l Layers) Cleargrads() {
	for k := range l {
		l[k].Cleargrads()
	}
}
