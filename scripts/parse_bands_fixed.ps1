$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

$xmlStart = 568
$sampleChunk = [System.Text.Encoding]::UTF8.GetString($bytes, $xmlStart, 30000)

Write-Output "=== XML SEARCH RESULTS ==="
$lines = $sampleChunk -split "`r?`n"
foreach ($line in $lines) {
    if ($line -match "BAND_NAME|POLARISATION|polarisation|DATA_TYPE|UNIT|DESCRIPTION|SOLAR_FLUX|INCIDENCE_ANGLE") {
        Write-Output $line.Trim()
    }
}
