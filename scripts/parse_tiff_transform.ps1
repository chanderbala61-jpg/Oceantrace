$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

# BitsPerSample Tag: count=2, type=3 (SHORT, 2 bytes). Tag entry at firstIfdOffset(8) + 2 + 2*12 = 34
# Entry bytes:
$e34 = $bytes[32..43]
Write-Output "BitsPerSample tag bytes: $($e34 -join ' ')"
$eBitsVal = [System.BitConverter]::ToUInt16($bytes, 40)
Write-Output "Val field bytes: $($bytes[40..43] -join ' ')"

# SampleFormat Tag entry is at entry index 13: 8 + 2 + 13*12 = 166
$e166 = $bytes[166..177]
Write-Output "SampleFormat tag bytes: $($e166 -join ' ')"

# Check Tag 34264 (ModelTransformationTag) at offset 376 (16 doubles = 128 bytes)
Write-Output "`n--- ModelTransformationTag (Affine matrix) at 376 ---"
for ($r = 0; $r -lt 4; $r++) {
    $rowVals = @()
    for ($c = 0; $c -lt 4; $c++) {
        $idx = ($r * 4 + $c)
        $offset = 376 + ($idx * 8)
        $dBytes = $bytes[$offset..($offset+7)]
        if ([System.BitConverter]::IsLittleEndian) { [Array]::Reverse($dBytes) }
        $d = [System.BitConverter]::ToDouble($dBytes, 0)
        $rowVals += "{0:E6}" -f $d
    }
    Write-Output "Row $r : $($rowVals -join '  ')"
}
