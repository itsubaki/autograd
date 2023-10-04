package variable

type config struct {
	EnableBackprop bool
	Train          bool
}

var Config = config{
	EnableBackprop: true,
	Train:          true,
}

func NoBackprop() func() {
	Config.EnableBackprop = false
	return func() {
		Config.EnableBackprop = true
	}
}

func TestMode() func() {
	Config.Train = false
	return func() {
		Config.Train = true
	}
}
