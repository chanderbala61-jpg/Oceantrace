$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

function Read-UInt16($offset) {
    return ([uint16]$bytes[$offset] -shl 8) -bor [uint16]$bytes[$offset+1]
}
function Read-UInt32($offset) {
    return ([uint32]$bytes[$offset] -shl 24) -bor ([uint32]$bytes[$offset+1] -shl 16) -bor ([uint32]$bytes[$offset+2] -shl 8) -bor [uint32]$bytes[$offset+3]
}

Write-Output "--- BitsPerSample & SampleFormat ---"
# BitsPerSample count 2 at offset 226
$b1 = Read-UInt16 226
$b2 = Read-UInt16 228
Write-Output "BitsPerSample: $b1, $b2"

# SampleFormat count 2 at offset 230
$f1 = Read-UInt16 230
$f2 = Read-UInt16 232
Write-Output "SampleFormat: $f1, $f2 (1=uint, 2=int, 3=ieee_fp, 4=void)"

Write-Output "`n--- GeoAsciiParams (15 chars) at 552 ---"
$asciiStr = [System.Text.Encoding]::ASCII.GetString($bytes, 552, 15)
Write-Output "GeoAscii: $asciiStr"

Write-Output "`n--- GeoKeyDirectoryTag at 504 ---"
for ($k=0; $k -lt 24; $k++) {
    $v = Read-UInt16 (504 + $k*2)
    Write-Output "Key[$k]: $v"
}

Write-Output "`n--- Tag 65000 (XML / Metadata) length 5659076 at 568 ---"
$headerChunk = [System.Text.Encoding]::UTF8.GetString($bytes, 568, [Math]::Min(1000, 5659076))
Write-Output "Tag 65000 start: $headerChunk"
