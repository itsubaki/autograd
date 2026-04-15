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
type Span struct {
	End func()
}

// Nograd disables backpropagation until End is called.
func Nograd() *Span {
	Config.EnableBackprop = false
	return &Span{
		End: func() {
			Config.EnableBackprop = true
		},
	}
}

// TestMode disables training mode until End is called.
func TestMode() *Span {
	Config.Train = false
	return &Span{
		End: func() {
			Config.Train = true
		},
	}
}
