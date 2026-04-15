package layer

import (
	"iter"
	"slices"

	"github.com/itsubaki/autograd/variable"
)

// Parameter is an alias for a trainable variable.
type Parameter = *variable.Variable

// Parameters is a named collection of parameters.
type Parameters map[string]Parameter

// Add stores a parameter under the given name and updates param.Name.
func (p Parameters) Add(name string, param Parameter) {
	param.Name = name
	p[name] = param
}

// Delete removes the parameter with the given name.
func (p Parameters) Delete(name string) {
	delete(p, name)
}

// Params returns the parameter collection itself.
func (p Parameters) Params() Parameters {
	return p
}

// Cleargrads clears gradients for all parameters in the collection.
func (p Parameters) Cleargrads() {
	for k := range p {
		p[k].Cleargrad()
	}
}

// Seq2 returns the parameters in key-sorted order.
func (p Parameters) Seq2() iter.Seq2[string, Parameter] {
	keys := make([]string, 0, len(p))
	for k := range p {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	return func(yield func(string, Parameter) bool) {
		for _, k := range keys {
			if !yield(k, p[k]) {
				return
			}
		}
	}
}
