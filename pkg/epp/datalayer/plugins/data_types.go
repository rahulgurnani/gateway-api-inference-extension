package plugins

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

const (
	PrefixCacheMatchPrecentKey = "PrefixCacheMatchPercentKey"
)

type PrefixCacheMatchPercent struct {
	matchPercentage float64
}

func NewPrefixCacheMatchPercent(matchPercentage float64) *PrefixCacheMatchPercent {
	return &PrefixCacheMatchPercent{
		matchPercentage: matchPercentage,
	}
}

func (p *PrefixCacheMatchPercent) MatchPercentage() float64 {
	return p.matchPercentage
}

func (p *PrefixCacheMatchPercent) Clone() datalayer.Cloneable {
	return &PrefixCacheMatchPercent{
		matchPercentage: p.matchPercentage,
	}
}
