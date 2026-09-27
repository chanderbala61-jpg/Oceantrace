$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

$str = [System.Text.Encoding]::UTF8.GetString($bytes, 568, 50000)
$lines = $str -split "`r?`n"
$inSpectralBand = $false
foreach ($line in $lines) {
    if ($line -like "*<Spectral_Band_Info>*") { $inSpectralBand = $true }
    if ($line -like "*</Spectral_Band_Info>*") { Write-Output $line.Trim(); $inSpectralBand = $false }
    if ($inSpectralBand) {
        Write-Output $line.Trim()
    }
}
