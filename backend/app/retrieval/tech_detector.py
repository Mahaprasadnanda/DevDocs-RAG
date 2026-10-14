import re
from typing import List, Optional

TECH_PATTERNS = {
    "spring-boot": [
        r"\bspring[-\s]?boot\b",
        r"\bspring\b",
        r"\borg\.springframework\b",
        r"@SpringBootApplication",
        r"@RestController",
        r"@Controller",
        r"@Autowired",
        r"@RequestMapping",
        r"@GetMapping",
        r"@PostMapping",
        r"@Configuration",
        r"@Component",
        r"@Service",
        r"@Repository",
        r"@ConfigurationProperties",
        r"\bApplicationContext\b",
        r"\bNoSuchBeanDefinitionException\b",
        r"\bBeanCreationException\b",
        r"\bapplication\.(?:properties|yml|yaml)\b",
        r"\bpom\.xml\b",
        r"\bmaven\b",
        r"\bgradle\b",
        r"\bResponseEntity\b",
        r"\bresilience4j\b",
        r"\bspring[-\s]?cloud\b",
        r"\bRequestRateLimiter\b",
        r"\bKeyResolver\b",
        r"\bBucket4j\b",
        r"@RateLimiter"
    ],
    "nodejs": [
        r"\bnode(?:\.js)?\b",
        r"\bERR_MODULE_NOT_FOUND\b",
        r"\bERR_REQUIRE_ESM\b",
        r"\bERR_INVALID_ARG_TYPE\b",
        r"\bMODULE_NOT_FOUND\b",
        r"\bpackage\.json\b",
        r"\bprocess\.env\b",
        r"\bprocess\.exit\b",
        r"\bhttp\.createServer\b",
        r"\bfs\.(?:readFile|writeFile|promises)\b",
        r"\bmodule\.exports\b",
        r"\brequire\(",
        r"\bnode:[a-z0-9_-]+\b",
        r"\bEventEmitter\b",
        r"\bBuffer\b"
    ],
    "go": [
        r"\bgolang\b",
        r"\bgo\.mod\b",
        r"\bgo\.sum\b",
        r"\bgo\s+(?:build|run|test|get)\b",
        r"\bnet/http\b",
        r"\bhttp\.Handler\b",
        r"\bhttp\.HandleFunc\b",
        r"\bhttp\.ResponseWriter\b",
        r"\bhttp\.ListenAndServe\b",
        r"\bgoroutine\b",
        r"\bchannel\b",
        r"\bpanic:\s",
        r"\bdefer\s",
        r"\bfmt\.Print",
        r"\bsync\.WaitGroup\b",
        r"\bfunc\s+[A-Za-z0-9_]+\s*\(",
        r"\btype\s+[A-Za-z0-9_]+\s+(?:struct|interface)\b"
    ],
    "react": [
        r"\breact(?:js|\.js)?\b",
        r"\buseState\b",
        r"\buseEffect\b",
        r"\buseContext\b",
        r"\buseReducer\b",
        r"\buseMemo\b",
        r"\buseCallback\b",
        r"\buseRef\b",
        r"\bJSX\b",
        r"\bTSX\b",
        r"\bprops\b",
        r"\bclassName=",
        r"\bReactDOM\b",
        r"\bcreateRoot\b",
        r"\bInvalid hook call\b",
        r"\bHooks can only be called\b",
        r"\bRendered more hooks\b"
    ],
    "django": [
        r"\bdjango\b",
        r"\bdjango\.urls\b",
        r"\bdjango\.db\b",
        r"\bmodels\.Model\b",
        r"\bforms\.Form\b",
        r"\bmanage\.py\b",
        r"\bsettings\.py\b",
        r"\bINSTALLED_APPS\b",
        r"\burlpatterns\b",
        r"\bDoesNotExist\b",
        r"\bHttpResponse\b",
        r"\bHttpRequest\b"
    ],
    "flask": [
        r"\bflask\b",
        r"\bFlask\(__name__\)",
        r"@app\.route",
        r"\brender_template\b",
        r"\bjsonify\b",
        r"\burl_for\b",
        r"\bwerkzeug\b",
        r"\brequest\.args\b",
        r"\brequest\.form\b",
        r"\bBlueprint\b"
    ],
    "pydantic": [
        r"\bpydantic\b",
        r"\bBaseModel\b",
        r"\bValidationError\b",
        r"\bField\(",
        r"\bfield_validator\b",
        r"\bmodel_validator\b",
        r"\bmodel_dump\b",
        r"\bmodel_validate\b"
    ],
    "fastapi": [
        r"\bfastapi\b",
        r"\bFastAPI\(\)",
        r"@app\.(?:get|post|put|delete|patch)\b",
        r"\bHTTPException\b",
        r"\buvicorn\b",
        r"\bDepends\(",
        r"\bPath\(",
        r"\bQuery\(",
        r"\bBody\("
    ]
}


def detect_technology(
    query: Optional[str] = None,
    context: Optional[str] = None,
    error_message: Optional[str] = None
) -> List[str]:
    """
    Detects the likely technology or technologies from input text.
    Returns a list of matching tech IDs (e.g. ['spring-boot'], ['react', 'spring-boot'], or [] if uncertain).
    """
    combined_parts = [p for p in [query, context, error_message] if p and p.strip()]
    if not combined_parts:
        return []

    combined_text = "\n".join(combined_parts)
    detected = []

    for tech_id, patterns in TECH_PATTERNS.items():
        score = 0
        for pat in patterns:
            if re.search(pat, combined_text, re.IGNORECASE):
                score += 1
        if score > 0:
            detected.append((tech_id, score))

    if not detected:
        return []

    # Sort by number of matching patterns descending
    detected.sort(key=lambda x: x[1], reverse=True)
    max_score = detected[0][1]

    # Return top technologies (support cross-technology when multiple are strong)
    result = [t[0] for t in detected if t[1] >= max(1, max_score - 1)]
    return result
