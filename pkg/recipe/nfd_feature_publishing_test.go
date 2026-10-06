// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package recipe

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const (
	nfdPubNFD             = "nfd"
	nfdPubNetworkOperator = "network-operator"
	nfdPubMellanoxLabel   = "feature.node.kubernetes.io/pci-15b3.present"
	nfdPubMellanoxBare    = "pci-15b3.present"
)

// nfdPubChartRule is a NodeFeatureRule that a registry chart renders itself,
// outside recipes/components/*/manifests. nfd-master evaluates it against the
// features nfd-worker publishes, so the feature it reads must not appear in
// worker.config.core.noPublishFeatures.
type nfdPubChartRule struct {
	component    string
	enablePath   []string
	chartDefault bool
	feature      string
}

// nfdPubChartRules comes from the chart templates at the registry pins on
// 2026-10-06: network-operator 26.4.1 templates/nodefeaturerules.yaml,
// gpu-operator v26.7.1 templates/nodefeaturerules.yaml, k8s-nim-operator 3.1.0
// templates/node-feature-rule.yaml. Re-check it when one of those pins moves.
var nfdPubChartRules = []nfdPubChartRule{
	{component: "network-operator", enablePath: []string{"nfd", "deployNodeFeatureRules"}, chartDefault: true, feature: "pci.device"},
	{component: "gpu-operator", enablePath: []string{"nfd", "nodefeaturerules"}, chartDefault: false, feature: "kernel.loadedmodule"},
	{component: "k8s-nim-operator", enablePath: []string{"nfd", "nodeFeatureRules", "deviceID"}, chartDefault: false, feature: "pci.device"},
	{component: "k8s-nim-operator-ocp", enablePath: []string{"nfd", "nodeFeatureRules", "deviceID"}, chartDefault: false, feature: "pci.device"},
}

// nfdPubTransitional is the expand phase of the #3038 pci-15b3.present
// handoff: for one release the cluster-side NIC rules ship beside the
// nfd-worker rule, so the label never drops while nfd-worker rolls, and once a
// node's worker strips pci.* they match nothing. Each key is a chart rule
// (component and enable path) or a manifest path; the value is the feature it
// reads. An entry is skipped only on a leaf whose nfd-worker rule exists
// (invariant 1). Delete the entries together with the rules in the contract
// change.
var nfdPubTransitional = map[string]string{
	"network-operator nfd.deployNodeFeatureRules":                 "pci.device",
	"components/network-operator/manifests/nfd-network-rule.yaml": "pci.device",
}

// nfdPubStripList is the #3038 nfd-master memory fix and must change together
// with worker.config.core.noPublishFeatures in recipes/components/nfd/values.yaml.
var nfdPubStripList = []string{
	"cpu.*", "kernel.config", "kernel.enabledmodule", "kernel.kvm", "kernel.selinux", "kernel.version",
	"local.*", "memory.*", "network.*", "pci.*", "storage.*", "system.*", "usb.*",
}

var nfdPubManifestFeatureRe = regexp.MustCompile(`(?m)^\s*(?:-\s*)?feature:\s*["']?([A-Za-z0-9_.]+)["']?\s*$`)

// TestNFDFeaturePublishingContract resolves every shipped leaf and checks, on
// merged effective values:
//
//  1. A recipe with network-operator has exactly one nfd-worker sources.custom
//     rule producing pci-15b3.present="true" (the NicClusterPolicy
//     nodeAffinity and the RDMA readiness gate cohort), matching vendor 15b3
//     and device 101c/101e on AKS, vendor 15b3 and class 0200/0207 elsewhere.
//  2. A recipe without network-operator has no such rule, whatever its value:
//     the GPU Operator validator waits for the network-operator MOFED driver
//     on labeled nodes when GPUDirect RDMA is on and useHostMofed is false.
//  3. No NodeFeatureRule the recipe renders (chart-shipped or a manifest)
//     reads a feature that noPublishFeatures strips, except the two
//     transitional NIC rules (see nfdPubTransitional).
//  4. A recipe with nfd strips exactly nfdPubStripList: the nfd-master memory
//     fix for #3038.
//  5. A recipe with nfd has no worker.config.sources.pci that emits
//     pci-15b3.present, a second producer invariants 1 and 2 cannot see.
//  6. A recipe with network-operator leaves the rule in 1 running: nothing
//     disables the custom label source or the pci feature source, and no
//     label whitelist filters labels.
//
// A rule label counts with or without the feature.node.kubernetes.io/ prefix,
// because nfd-master adds that prefix to an un-namespaced label.
func TestNFDFeaturePublishingContract(t *testing.T) {
	ctx := context.Background()
	store, err := buildMetadataStore(ctx, defaultEmbeddedProvider)
	if err != nil {
		t.Fatalf("buildMetadataStore: %v", err)
	}

	leaves := 0
	for name, overlay := range store.Overlays {
		if overlay.Spec.Criteria == nil {
			continue
		}
		leaves++
		criteria := overlay.Spec.Criteria
		t.Run(name, func(t *testing.T) {
			result, err := store.BuildRecipeResult(ctx, criteria)
			if err != nil {
				t.Fatalf("resolve %s: %v", name, err)
			}
			if !slices.Contains(result.Metadata.AppliedOverlays, name) {
				t.Fatalf("%s: criteria resolved to %v, which does not include it", name, result.Metadata.AppliedOverlays)
			}

			hasNFD := nfdPubEnabled(result.ComponentRefs, nfdPubNFD)
			hasNetOp := nfdPubEnabled(result.ComponentRefs, nfdPubNetworkOperator)

			var patterns []string
			var rules []map[string]any
			if hasNFD {
				values, err := result.GetValuesForComponentWithContext(ctx, nfdPubNFD)
				if err != nil {
					t.Fatalf("%s: nfd values: %v", name, err)
				}
				patterns = nfdPubStrings(nfdPubLookup(values, "worker", "config", "core", "noPublishFeatures"))
				rules = nfdPubMellanoxRules(values)
				if whitelist, fields, ok := nfdPubPCISourceNIC(values); ok {
					t.Errorf("%s: nfd worker.config.sources.pci labels Mellanox NICs %s (effective deviceLabelFields %q, "+
						"deviceClassWhitelist %q): a second producer bypasses invariants 1 and 2, and on AKS labels the "+
						"accelerated-networking Ethernet VFs", name, nfdPubMellanoxLabel, fields, whitelist)
				}
				if hasNetOp {
					for _, problem := range nfdPubWorkerRuleProblems(values) {
						t.Errorf("%s: %s, so the nfd-worker rule labeling %s never runs", name, problem, nfdPubMellanoxLabel)
					}
				}
			}

			switch {
			case hasNetOp && !hasNFD:
				t.Fatalf("%s ships network-operator without the nfd component, so nothing labels %s", name, nfdPubMellanoxLabel)
			case hasNetOp:
				if len(rules) != 1 {
					t.Errorf("%s ships network-operator: want exactly one nfd-worker sources.custom rule labeling %s (or %s), got %d; "+
						"set the nfd componentRef valuesFile to components/nfd/values-nvidia-nics.yaml (values-nvidia-nics-aks.yaml on AKS)",
						name, nfdPubMellanoxLabel, nfdPubMellanoxBare, len(rules))
				} else {
					for _, problem := range nfdPubNICRuleProblems(rules[0], criteria.Service == CriteriaServiceAKS) {
						t.Errorf("%s: %s", name, problem)
					}
				}
			default:
				if len(rules) != 0 {
					t.Errorf("%s has no network-operator but nfd-worker labels %s (%d rules): the GPU Operator validator would wait "+
						"for a MOFED driver nothing installs", name, nfdPubMellanoxLabel, len(rules))
				}
			}

			if hasNFD && !slices.Equal(patterns, nfdPubStripList) {
				file := "recipes/components/nfd/values.yaml"
				if ref, _ := findComponentRefByName(result.ComponentRefs, nfdPubNFD); ref.ValuesFile != "" && ref.ValuesFile != "components/nfd/values.yaml" {
					file += ", or recipes/" + ref.ValuesFile + " if its list replaced it"
				}
				t.Errorf("%s: nfd worker.config.core.noPublishFeatures = %q, want %q (nfdPubStripList); change %s",
					name, patterns, nfdPubStripList, file)
			}

			if len(patterns) == 0 {
				return
			}
			redundant := hasNetOp && len(rules) == 1
			for _, cr := range nfdPubChartRules {
				if !nfdPubEnabled(result.ComponentRefs, cr.component) {
					continue
				}
				values, err := result.GetValuesForComponentWithContext(ctx, cr.component)
				if err != nil {
					t.Fatalf("%s: %s values: %v", name, cr.component, err)
				}
				enabled := cr.chartDefault
				if v, ok := nfdPubLookup(values, cr.enablePath...).(bool); ok {
					enabled = v
				}
				transitional := redundant && nfdPubTransitional[cr.component+" "+strings.Join(cr.enablePath, ".")] == cr.feature
				if enabled && nfdPubStripped(cr.feature, patterns) && !transitional {
					t.Errorf("%s: %s renders a NodeFeatureRule (%s) reading %s, which nfd.worker.config.core.noPublishFeatures strips",
						name, cr.component, strings.Join(cr.enablePath, "."), cr.feature)
				}
			}
			for _, ref := range result.ComponentRefs {
				if !ref.IsEnabled() {
					continue
				}
				for _, p := range slices.Concat(ref.PreManifestFiles, ref.ManifestFiles) {
					content, err := GetManifestContentWithContext(ctx, nil, p)
					if err != nil {
						t.Fatalf("%s: read %s: %v", name, p, err)
					}
					for _, f := range nfdPubManifestRuleFeatures(string(content)) {
						transitional := redundant && nfdPubTransitional[p] == f
						if nfdPubStripped(f, patterns) && !transitional {
							t.Errorf("%s: manifest %s has a NodeFeatureRule reading %s, which noPublishFeatures strips", name, p, f)
						}
					}
				}
			}
		})
	}
	if leaves == 0 {
		t.Fatal("no leaf overlays found; the overlay walker is broken")
	}
}

// TestNFDPubStripped pins the pattern semantics to nfd-worker v0.19.0
// patternMatches: exact key, or prefix when the pattern ends in "*".
func TestNFDPubStripped(t *testing.T) {
	tests := []struct {
		name     string
		feature  string
		patterns []string
		want     bool
	}{
		{"exact match", "kernel.config", []string{"kernel.config"}, true},
		{"exact pattern does not prefix-match", "kernel.configs", []string{"kernel.config"}, false},
		{"prefix pattern", "pci.device", []string{"pci.*"}, true},
		{"sibling key kept", "kernel.loadedmodule", []string{"kernel.config", "kernel.enabledmodule"}, false},
		{"no patterns", "pci.device", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nfdPubStripped(tt.feature, tt.patterns); got != tt.want {
				t.Errorf("nfdPubStripped(%q, %v) = %v, want %v", tt.feature, tt.patterns, got, tt.want)
			}
		})
	}
}

func nfdPubEnabled(refs []ComponentRef, name string) bool {
	ref, ok := findComponentRefByName(refs, name)
	return ok && ref.IsEnabled()
}

func nfdPubLookup(m map[string]any, path ...string) any {
	var cur any = m
	for _, k := range path {
		mm, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = mm[k]
	}
	return cur
}

func nfdPubStrings(v any) []string {
	list, _ := v.([]any)
	out := make([]string, 0, len(list))
	for _, e := range list {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func nfdPubStripped(feature string, patterns []string) bool {
	for _, p := range patterns {
		if prefix, ok := strings.CutSuffix(p, "*"); ok {
			if strings.HasPrefix(feature, prefix) {
				return true
			}
		} else if feature == p {
			return true
		}
	}
	return false
}

func nfdPubMellanoxRules(values map[string]any) []map[string]any {
	var out []map[string]any
	list, _ := nfdPubLookup(values, "worker", "config", "sources", "custom").([]any)
	for _, e := range list {
		rule, ok := e.(map[string]any)
		if !ok {
			continue
		}
		labels, _ := rule["labels"].(map[string]any)
		_, prefixed := labels[nfdPubMellanoxLabel]
		_, bare := labels[nfdPubMellanoxBare]
		if prefixed || bare {
			out = append(out, rule)
		}
	}
	return out
}

// nfdPubPCISourceNIC returns the effective nfd-worker sources.pci lists and
// whether they label a Mellanox NIC (class 0200 or 0207) pci-15b3.present.
// Per NFD v0.19.0 source/pci/pci.go: the defaults are whitelist
// ["03","0b40","12"] and fields ["class","vendor"] (:46-51); configured fields
// are kept in mandatoryDevAttrs order (utils.go:31) with unknown names dropped,
// falling back to the defaults when none is left (:95-113); a device is
// labeled when a whitelist entry prefixes its 4-digit class, and the label
// joins the fields with "_" (:116-128), so only ["vendor"] yields 15b3.present.
func nfdPubPCISourceNIC(values map[string]any) (whitelist, fields []string, labels bool) {
	pci, _ := nfdPubLookup(values, "worker", "config", "sources", "pci").(map[string]any)
	whitelist = []string{"03", "0b40", "12"}
	if v := pci["deviceClassWhitelist"]; v != nil {
		whitelist = nfdPubStrings(v)
	}
	configured := []string{"class", "vendor"}
	if v := pci["deviceLabelFields"]; v != nil {
		configured = nfdPubStrings(v)
	}
	for _, attr := range []string{"class", "vendor", "device", "subsystem_vendor", "subsystem_device"} {
		if slices.Contains(configured, attr) {
			fields = append(fields, attr)
		}
	}
	if len(fields) == 0 {
		fields = []string{"class", "vendor"}
	}
	if !slices.Equal(fields, []string{"vendor"}) {
		return whitelist, fields, false
	}
	for _, w := range whitelist {
		for _, class := range []string{"0200", "0207"} {
			if strings.HasPrefix(class, strings.ToLower(w)) {
				return whitelist, fields, true
			}
		}
	}
	return whitelist, fields, false
}

// nfdPubWorkerRuleProblems lists the merged nfd values that would keep the
// nfd-worker sources.custom rule from labeling a node. Per NFD v0.19.0
// pkg/nfd-worker/nfd-worker.go: labelSources and featureSources default to
// ["all"] (:302-303), the custom label source reads features only from
// enabled feature sources (:340, source/source.go:182), the deprecated
// core.sources replaces labelSources (:734-736), and labelWhiteList filters
// every label source (:792-836). The worker flags -label-sources,
// -feature-sources and -options override the config file (:672-691,
// :744-764); cmd/nfd-worker/main.go:106-148 has no -label-whitelist flag.
func nfdPubWorkerRuleProblems(values map[string]any) []string {
	var problems []string
	core, _ := nfdPubLookup(values, "worker", "config", "core").(map[string]any)
	if v := core["sources"]; v != nil {
		problems = append(problems, fmt.Sprintf("nfd worker.config.core.sources is %v (deprecated; it replaces labelSources)", v))
	}
	if v := core["labelSources"]; v != nil && !nfdPubSourceEnabled(nfdPubStrings(v), "custom") {
		problems = append(problems, fmt.Sprintf("nfd worker.config.core.labelSources %v disables the custom label source", v))
	}
	if v := core["featureSources"]; v != nil && !nfdPubSourceEnabled(nfdPubStrings(v), "pci") {
		problems = append(problems, fmt.Sprintf("nfd worker.config.core.featureSources %v disables the pci feature source", v))
	}
	if v := core["labelWhiteList"]; v != nil && v != "" {
		problems = append(problems, fmt.Sprintf("nfd worker.config.core.labelWhiteList is %#v", v))
	}
	args, _ := nfdPubLookup(values, "worker", "extraArgs").([]any)
	for _, a := range args {
		s, _ := a.(string)
		if !strings.HasPrefix(s, "-") {
			continue
		}
		for _, flag := range []string{"label-sources", "feature-sources", "options"} {
			if strings.HasPrefix(strings.TrimLeft(s, "-"), flag) {
				problems = append(problems, fmt.Sprintf("nfd worker.extraArgs passes %q, which overrides the config file", s))
			}
		}
	}
	return problems
}

// nfdPubSourceEnabled replays nfd-worker v0.19.0 source-list processing
// (nfd-worker.go:572-598, :604-630): entries apply in order, "all" enables
// every source (none but the fake source is off by default), "-name"
// disables one.
func nfdPubSourceEnabled(list []string, name string) bool {
	on := false
	for _, s := range list {
		switch s {
		case "all", name:
			on = true
		case "-" + name:
			on = false
		}
	}
	return on
}

// nfdPubNICRuleProblems compares a pci-15b3.present rule against the
// values-nvidia-nics*.yaml contract. On AKS a class match would also label
// the Mellanox Ethernet VFs that Azure accelerated networking attaches, so
// AKS matches the InfiniBand VF device IDs instead.
func nfdPubNICRuleProblems(rule map[string]any, aks bool) []string {
	var problems []string
	labels, _ := rule["labels"].(map[string]any)
	for _, key := range []string{nfdPubMellanoxLabel, nfdPubMellanoxBare} {
		if v, ok := labels[key]; ok && v != "true" {
			problems = append(problems, fmt.Sprintf("rule %v label %s: got %#v, want the string \"true\"", rule["name"], key, v))
		}
	}

	exprs := nfdPubPCIExpressions(rule)
	want := []struct {
		key    string
		values []string
	}{
		{"vendor", []string{"15b3"}},
		{"class", []string{"0200", "0207"}},
	}
	forbidden := "device"
	if aks {
		want[1].key, want[1].values = "device", []string{"101c", "101e"}
		forbidden = "class"
	}
	for _, w := range want {
		op, values, ok := nfdPubExpression(exprs, w.key)
		if !ok {
			problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got no expression, want op \"In\" value %q",
				rule["name"], w.key, w.values))
			continue
		}
		if op != "In" || !slices.Equal(values, w.values) {
			problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got op %q value %q, want op \"In\" value %q",
				rule["name"], w.key, op, values, w.values))
		}
	}
	if _, ok := exprs[forbidden]; ok {
		problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got an expression, want none (aks=%v)",
			rule["name"], forbidden, aks))
	}
	return problems
}

// nfdPubExpression decodes one matchExpressions entry, {op: In, value: [...]}.
// A non-string value element is rendered with its type so it never equals a
// wanted ID.
func nfdPubExpression(exprs map[string]any, key string) (string, []string, bool) {
	e, ok := exprs[key].(map[string]any)
	if !ok {
		return "", nil, false
	}
	op, _ := e["op"].(string)
	list, _ := e["value"].([]any)
	values := make([]string, 0, len(list))
	for _, v := range list {
		if s, ok := v.(string); ok {
			values = append(values, s)
		} else {
			values = append(values, fmt.Sprintf("%T(%v)", v, v))
		}
	}
	return op, values, true
}

func nfdPubPCIExpressions(rule map[string]any) map[string]any {
	matchers, _ := rule["matchFeatures"].([]any)
	for _, m := range matchers {
		mm, ok := m.(map[string]any)
		if !ok || mm["feature"] != "pci.device" {
			continue
		}
		exprs, _ := mm["matchExpressions"].(map[string]any)
		return exprs
	}
	return nil
}

func nfdPubManifestRuleFeatures(content string) []string {
	var out []string
	for _, doc := range strings.Split(content, "\n---") {
		if !strings.Contains(doc, "kind: NodeFeatureRule") && !strings.Contains(doc, "kind: NodeFeatureGroup") {
			continue
		}
		for _, m := range nfdPubManifestFeatureRe.FindAllStringSubmatch(doc, -1) {
			out = append(out, m[1])
		}
	}
	return out
}
