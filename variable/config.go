package variable

type config struct {
	EnableBackprop bool
	Train          bool
}

var Config = config{
	EnableBackprop: true,
	Train:          true,
}

// Span represents a temporary configuration scope.
// Scopes should be ended in reverse order of creation.
type Span struct {
	End func()
}

// Nograd disables backpropagation until End restores the previous setting.
func Nograd() *Span {
	previous := Config.EnableBackprop
	Config.EnableBackprop = false
	return &Span{
		End: func() {
			Config.EnableBackprop = previous
		},
	}
}

// TestMode disables training mode until End restores the previous setting.
func TestMode() *Span {
	previous := Config.Train
	Config.Train = false
	return &Span{
		End: func() {
			Config.Train = previous
		},
	}
}
