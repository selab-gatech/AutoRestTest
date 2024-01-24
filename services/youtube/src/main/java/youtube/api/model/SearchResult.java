package youtube.api.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonPropertyOrder({"kind", "etag", "id", "snippet"})
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SearchResult {

    private String etag;

    private ResourceId id;

    private String kind;

    private SearchResultSnippet snippet;

    public String getEtag() {
        return etag;
    }

    public void setEtag(String etag) {
        this.etag = etag;
    }

    public ResourceId getId() {
        return id;
    }

    public void setId(ResourceId id) {
        this.id = id;
    }

    public String getKind() {
        return kind;
    }

    public void setKind(String kind) {
        this.kind = kind;
    }

    public SearchResultSnippet getSnippet() {
        return snippet;
    }

    public void setSnippet(SearchResultSnippet snippet) {
        this.snippet = snippet;
    }
}
