$ErrorActionPreference = "Stop"

$mockDir = "data/mock"
New-Item -ItemType Directory -Path $mockDir -Force | Out-Null

$csvPath = Join-Path $mockDir "profiles_100_db_like.csv"
$reqPath = Join-Path $mockDir "profiles_100_requests.jsonl"
$resPath = Join-Path $mockDir "profiles_100_results.jsonl"
$sumPath = Join-Path $mockDir "profiles_100_summary.json"

$firstNames = @("Yassine", "Sara", "Hamza", "Aya", "Mehdi", "Salma", "Omar", "Nour", "Anas", "Lina")
$lastNames = @("Bennani", "Alaoui", "Idrissi", "Amrani", "Tazi", "Chraibi", "Berrada", "Fassi", "ElMansouri", "Skalli")
$cities = @("Casablanca", "Rabat", "Marrakech", "Fes", "Tanger", "Agadir", "Meknes", "Oujda", "Kenitra", "Tetouan")
$series = @("Sciences Mathematiques", "Sciences Physiques", "Sciences de la Vie et de la Terre", "Sciences Economiques", "Lettres", "Arts")
$grades = @("passable", "bien", "tres_bien", "elite")
$motivations = @("cash", "prestige", "passion", "safety", "expat", "employability")
$budgets = @("0dh", "serre", "confort", "illimite")
$classes = @("2eme-bac", "1ere-bac", "bac-plus")

$baseDate = Get-Date "2026-01-01T09:00:00"

$rows = New-Object System.Collections.Generic.List[object]

for ($i = 1; $i -le 100; $i++) {
    $first = $firstNames[($i - 1) % $firstNames.Count]
    $last = $lastNames[[int](($i - 1) / $firstNames.Count) % $lastNames.Count]
    $city = $cities[($i - 1) % $cities.Count]
    $serie = $series[($i - 1) % $series.Count]
    $grade = $grades[($i - 1) % $grades.Count]
    $motivation = $motivations[($i - 1) % $motivations.Count]
    $budget = $budgets[($i - 1) % $budgets.Count]
    $classe = $classes[($i - 1) % $classes.Count]

    $created = $baseDate.AddDays($i).AddMinutes($i * 7)
    $updated = $created.AddDays($i % 19).AddMinutes(13)

    $digits = "{0:D8}" -f (12000000 + ($i * 173))
    $phone = "+212 6 {0} {1} {2} {3}" -f $digits.Substring(0, 2), $digits.Substring(2, 2), $digits.Substring(4, 2), $digits.Substring(6, 2)

    $rows.Add([pscustomobject][ordered]@{
        id = [guid]::NewGuid().Guid
        name = "$first $last"
        stripeCustomerId = ""
        paypalCustomerIds = "[]"
        email = ("student{0:D3}@example.ma" -f $i)
        phone = $phone
        whatsapp = ""
        ville = $city
        serie_bac = $serie
        note_esperee = $grade
        motivation = $motivation
        budget = $budget
        role = "CLIENT"
        status = $(if ($i % 7 -eq 0) { "INACTIVE" } else { "ACTIVE" })
        languages = $(if ($i % 5 -eq 0) { '["fr","en"]' } else { '["fr"]' })
        isSeller = $false
        activeRole = "CLIENT"
        password = ""
        deletedAt = ""
        emailVerified = $true
        trialled = $false
        referralCode = $(if ($i % 9 -eq 0) { "REF{0:D3}" -f $i } else { "" })
        age = 17 + ($i % 10)
        mediaId = ""
        createdAt = $created.ToString("yyyy-MM-dd HH:mm:ss.fff")
        updatedAt = $updated.ToString("yyyy-MM-dd HH:mm:ss.fff")
        country = "MA"
        classe = $classe
    }) | Out-Null
}

$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding utf8

function Map-Bac([string]$value) {
    $v = ([string]$value).ToLower().Trim()
    if ($v -match "math") { return "sm" }
    if ($v -match "phys") { return "spc" }
    if ($v -match "vie|terre|svt") { return "svt" }
    if ($v -match "eco") { return "eco" }
    if ($v -match "lett") { return "lettres" }
    if ($v -match "art|design") { return "arts" }
    return "spc"
}

function Map-Grade([string]$value) {
    $v = ([string]$value).ToLower().Trim()
    if ($v -match "elite|excellent|18|19|20") { return "elite" }
    if ($v -match "tres_bien|tr[eè]s bien|16|17") { return "tres_bien" }
    if ($v -match "bien|14|15") { return "bien" }
    if ($v -match "passable|10|11|12") { return "passable" }
    return "bien"
}

function Map-Motivation([string]$value) {
    $v = ([string]$value).ToLower().Trim()
    switch ($v) {
        "cash" { return "cash" }
        "prestige" { return "prestige" }
        "passion" { return "passion" }
        "safety" { return "safety" }
        "expat" { return "expat" }
        "employability" { return "employability" }
        default { return "safety" }
    }
}

function Map-Budget([string]$value) {
    $v = ([string]$value).ToLower().Trim()
    if ($v -match "^0dh$|zero|public") { return "zero_public" }
    if ($v -match "serre|tight") { return "tight_25k" }
    if ($v -match "confort|comfort") { return "comfort_50k" }
    if ($v -match "illimite|no_limit") { return "no_limit_70k_plus" }
    return "comfort_50k"
}

$requestLines = New-Object System.Collections.Generic.List[string]

for ($idx = 0; $idx -lt $rows.Count; $idx++) {
    $row = $rows[$idx]
    $rid = "mock_{0:D3}" -f ($idx + 1)

    $req = [ordered]@{
        id = $rid
        question = ""
        top_k = 3
        profile = [ordered]@{
            bac_stream = Map-Bac $row.serie_bac
            expected_grade_band = Map-Grade $row.note_esperee
            motivation = Map-Motivation $row.motivation
            budget_band = Map-Budget $row.budget
            city = [string]$row.ville
            country = [string]$row.country
        }
        source_profile_id = [string]$row.id
    }

    $requestLines.Add(($req | ConvertTo-Json -Depth 8 -Compress)) | Out-Null
}

Set-Content -Path $reqPath -Value $requestLines -Encoding utf8

$results = New-Object System.Collections.Generic.List[object]
$ok = 0
$failed = 0

foreach ($line in $requestLines) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    $req = $line | ConvertFrom-Json
    $bodyObj = [ordered]@{
        question = [string]$req.question
        profile = $req.profile
        top_k = [int]$req.top_k
    }

    try {
        $resp = Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:3001/chat/query" -ContentType "application/json" -Body ($bodyObj | ConvertTo-Json -Depth 8)

        $topName = $null
        $rankedCount = 0
        if ($resp.ranked_schools) {
            $rankedCount = @($resp.ranked_schools).Count
            if ($rankedCount -gt 0) {
                $topName = [string]$resp.ranked_schools[0].name
                if ([string]::IsNullOrWhiteSpace($topName)) {
                    $topName = [string]$resp.ranked_schools[0].school_name
                }
            }
        }

        $results.Add([pscustomobject]@{
            id = [string]$req.id
            source_profile_id = [string]$req.source_profile_id
            profile = $req.profile
            confidence = [double]$resp.confidence
            message_paragraph = [string]$resp.message_paragraph
            top1_school = $topName
            ranked_count = $rankedCount
            status = "ok"
            error = ""
        }) | Out-Null

        $ok++
    }
    catch {
        $results.Add([pscustomobject]@{
            id = [string]$req.id
            source_profile_id = [string]$req.source_profile_id
            profile = $req.profile
            confidence = 0
            message_paragraph = ""
            top1_school = $null
            ranked_count = 0
            status = "error"
            error = [string]$_.Exception.Message
        }) | Out-Null

        $failed++
    }
}

$results | ForEach-Object { $_ | ConvertTo-Json -Depth 10 -Compress } | Set-Content -Path $resPath -Encoding utf8

$okResults = @($results | Where-Object { $_.status -eq "ok" })
$withRanked = @($okResults | Where-Object { [int]$_.ranked_count -gt 0 }).Count
$avgConf = 0
if ($okResults.Count -gt 0) {
    $avgConf = [math]::Round((($okResults | Measure-Object -Property confidence -Average).Average), 4)
}

$topDist = @(
    $okResults |
    Group-Object top1_school |
    Sort-Object Count -Descending |
    Select-Object -First 10 @{Name = "school"; Expression = { if ([string]::IsNullOrWhiteSpace($_.Name)) { "<none>" } else { $_.Name } } }, Count
)

$budgetBreakdown = @(
    $okResults |
    Group-Object { $_.profile.budget_band } |
    Sort-Object Name |
    ForEach-Object {
        $grp = @($_.Group)
        $avg = 0
        if ($grp.Count -gt 0) {
            $avg = [math]::Round((($grp | Measure-Object -Property confidence -Average).Average), 4)
        }

        [pscustomobject]@{
            budget_band = $(if ($_.Name) { $_.Name } else { "<none>" })
            count = $grp.Count
            avg_confidence = $avg
            with_ranked = @($grp | Where-Object { [int]$_.ranked_count -gt 0 }).Count
        }
    }
)

$cityBreakdown = @(
    $okResults |
    Group-Object { $_.profile.city } |
    Sort-Object Count -Descending |
    Select-Object -First 10 @{Name = "city"; Expression = { $_.Name } }, Count
)

$summary = [ordered]@{
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    total_profiles = $rows.Count
    total_requests = $requestLines.Count
    ok = $ok
    failed = $failed
    avg_confidence = $avgConf
    with_ranked = $withRanked
    without_ranked = ($okResults.Count - $withRanked)
    top1_distribution = $topDist
    budget_breakdown = $budgetBreakdown
    city_breakdown_top10 = $cityBreakdown
    files = [ordered]@{
        profiles_csv = $csvPath
        requests_jsonl = $reqPath
        results_jsonl = $resPath
        summary_json = $sumPath
    }
}

$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $sumPath -Encoding utf8
$summary | ConvertTo-Json -Depth 10
