$ErrorActionPreference = "Stop"

$samples = @("00000.tif", "00002.tif", "00642.tif")

foreach ($name in $samples) {
    $img_path = "extracted_dataset\images\$name"
    $msk_path = "extracted_dataset\masks\$name"
    
    $img_bytes = [System.IO.File]::ReadAllBytes($img_path)
    $msk_bytes = [System.IO.File]::ReadAllBytes($msk_path)
    
    Write-Output "=== SAMPLE $name ==="
    Write-Output "Image size on disk: $($img_bytes.Length) bytes ($([Math]::Round($img_bytes.Length/1MB, 2)) MB)"
    Write-Output "Mask size on disk:  $($msk_bytes.Length) bytes ($([Math]::Round($msk_bytes.Length/1MB, 2)) MB)"
    
    # Raster dimensions for uncompressed mask: 2048 x 2048 = 4194304 pixels
    # In uint8 uncompressed TIFF, header is 16780 bytes, followed by 4194304 bytes of pixel data
    $header_len = 16780
    $raster_len = 2048 * 2048
    $oil_count = 0
    $bg_count = 0
    $other_count = 0
    
    $unique_vals = New-Object 'System.Collections.Generic.HashSet[byte]'
    for ($i = $header_len; $i -lt ($header_len + $raster_len) -and $i -lt $msk_bytes.Length; $i++) {
        $v = $msk_bytes[$i]
        [void]$unique_vals.Add($v)
        if ($v -eq 0) { $bg_count++ }
        elseif ($v -eq 1) { $oil_count++ }
        else { $other_count++ }
    }
    
    Write-Output "Mask Unique Pixel Values: $($unique_vals -join ', ')"
    Write-Output "Background Pixels (0): $bg_count ($([Math]::Round($bg_count/$raster_len*100, 2))%)"
    Write-Output "Oil Spill Pixels (1):  $oil_count ($([Math]::Round($oil_count/$raster_len*100, 4))%)"
    Write-Output "Other Pixels (!= 0,1): $other_count"
    Write-Output ""
}
