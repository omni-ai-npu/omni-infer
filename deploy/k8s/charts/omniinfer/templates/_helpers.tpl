{{/*
Return the full image name in the format "repository:tag".
Usage: {{ include "omniinfer.image" (list .Values.image.repository .Values.image.tag) | quote }}
Parameters:
  - index 0: image repository (e.g. "registry-cbu.huawei.com/omniai_omniinfer/omni_infer-a3-arm")
  - index 1: image tag (e.g. "release_v0.5.0_20250924")
Defaults:
  - repository defaults to empty string
  - tag defaults to "latest"
*/}}
{{- define "omniinfer.image" -}}
{{- $repository := default "" (index . 0) -}}
{{- $tag := default "latest" (index . 1) -}}
{{- printf "%s:%s" $repository $tag -}}
{{- end -}}

{{/*
Return the replica count for a given role (e.g., "prefill" or "decode") from .Values.servingEngineSpec.modelSpec.
Usage:
  {{ include "omniinfer.replicaCountByRole" (dict "role" "prefill" "values" .Values) }}
*/}}
{{- define "omniinfer.replicaCountByRole" -}}
{{- $role := .role -}}
{{- $values := .values -}}
{{- $matchCount := 0 -}}
{{- $replicaCount := "0" -}}
{{- range $model := $values.servingEngineSpec.modelSpec -}}
  {{- if eq $model.role $role -}}
    {{- $matchCount = add $matchCount 1 -}}
    {{- if eq $matchCount 1 -}}
      {{- $replicaCount = printf "%.0f" $model.replicaCount -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- if gt $matchCount 1 -}}
  {{- fail (printf "错误：为角色 '%s' 找到 %d 个匹配项，但期望最多一个。" $role $matchCount) -}}
{{- else if eq $matchCount 1 -}}
  {{- $replicaCount -}}
{{- else -}}
  {{- "0" -}}
{{- end -}}
{{- end -}}

{{/*
Return the resources configuration for a given role (e.g., "prefill" or "decode") from .Values.servingEngineSpec.modelSpec.
Usage:
  {{ include "omniinfer.resourcesByRole" (dict "role" "prefill" "values" .Values) }}
*/}}
{{- define "omniinfer.resourcesByRole" -}}
{{- $role := .role -}}
{{- $values := .values -}}
{{- $resources := dict -}}
{{- range $model := $values.servingEngineSpec.modelSpec -}}
  {{- if eq $model.role $role -}}
    {{- $resources = $model.resources -}}
    {{- break -}} {{/* 取第一个匹配项 */}}
  {{- end -}}
{{- end -}}
{{- if $resources -}}
{{- toYaml $resources | nindent 10 -}}
{{- else -}}
{{- /* 如果没有找到匹配的role，返回空字典 */ -}}
{{- toYaml (dict) | nindent 10 -}}
{{- end -}}
{{- end -}}

{{/*
Return the apiPort for a given role (e.g., "prefill" or "decode") from .Values.servingEngineSpec.modelSpec.
Usage:
  {{ include "omniinfer.apiPortByRole" (dict "role" "decode" "values" .Values) }}
*/}}
{{- define "omniinfer.apiPortByRole" -}}
{{- $role := .role -}}
{{- $values := .values -}}
{{- $apiPort := "" -}}
{{- range $model := $values.servingEngineSpec.modelSpec -}}
  {{- if eq $model.role $role -}}
    {{- $apiPort = $model.apiPort -}}
    {{- break -}}
  {{- end -}}
{{- end -}}
{{- $apiPort -}}
{{- end -}}

{{/*
Return the servicePort for a given role (e.g., "prefill" or "decode") from .Values.servingEngineSpec.modelSpec.
Usage:
  {{ include "omniinfer.servicePortByRole" (dict "role" "decode" "values" .Values) }}
*/}}
{{- define "omniinfer.servicePortByRole" -}}
{{- $role := .role -}}
{{- $values := .values -}}
{{- $servicePort := "" -}}
{{- range $model := $values.servingEngineSpec.modelSpec -}}
  {{- if eq $model.role $role -}}
    {{- $servicePort = $model.servicePort -}}
    {{- break -}}
  {{- end -}}
{{- end -}}
{{- $servicePort -}}
{{- end -}}

{{- define "omniinfer.modelPathByRole" -}}
{{- $role := .role -}}
{{- $values := .values -}}
{{- $modelPath := "" -}}
{{- range $model := $values.servingEngineSpec.modelSpec -}}
  {{- if eq $model.role $role -}}
    {{- $modelPath = $model.modelPath -}}
    {{- break -}}
  {{- end -}}
{{- end -}}
{{- $modelPath -}}
{{- end -}}

{{- define "omniinfer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "omniinfer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "omniinfer.labels" -}}
helm.sh/chart: {{ include "omniinfer.chart" . }}
{{ include "omniinfer.selectorLabels" . }}
{{- if .Chart.Version }}
app.kubernetes.io/version: {{ .Chart.Version | quote }}
{{- end }}
app.kubernetes.io/managed-by: Helm
{{- end }}

{{/*
Selector labels
*/}}
{{- define "omniinfer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "omniinfer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}