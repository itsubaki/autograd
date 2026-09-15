package optimizer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/model"
)

var (
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.LSTM)(nil)
)

// Model is the interface implemented by optimizable models.
type Model interface {
	Params() layer.Parameters
}
