package youtube.api.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.List;

@JsonPropertyOrder({"kind", "etag", "nextPageToken", "prevPageToken", "regionCode", "pageInfo", "items", "eventId", "tokenPagination", "visitorId"})
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SearchListResponse {

    private String etag;

    private String eventId;

    private List<SearchResult> items;

    private String kind;

    private String nextPageToken;

    private PageInfo pageInfo;

    private String prevPageToken;

    private String regionCode;

    private TokenPagination tokenPagination;

    private String visitorId;

    public String getEtag() {
        return etag;
    }

    public void setEtag(String etag) {
        this.etag = etag;
    }

    public String getEventId() {
        return eventId;
    }

    public void setEventId(String eventId) {
        this.eventId = eventId;
    }

    public List<SearchResult> getItems() {
        return items;
    }

    public void setItems(List<SearchResult> items) {
        this.items = items;
    }

    public String getKind() {
        return kind;
    }

    public void setKind(String kind) {
        this.kind = kind;
    }

    public String getNextPageToken() {
        return nextPageToken;
    }

    public void setNextPageToken(String nextPageToken) {
        this.nextPageToken = nextPageToken;
    }

    public PageInfo getPageInfo() {
        return pageInfo;
    }

    public void setPageInfo(PageInfo pageInfo) {
        this.pageInfo = pageInfo;
    }

    public String getPrevPageToken() {
        return prevPageToken;
    }

    public void setPrevPageToken(String prevPageToken) {
        this.prevPageToken = prevPageToken;
    }

    public String getRegionCode() {
        return regionCode;
    }

    public void setRegionCode(String regionCode) {
        this.regionCode = regionCode;
    }

    public TokenPagination getTokenPagination() {
        return tokenPagination;
    }

    public void setTokenPagination(TokenPagination tokenPagination) {
        this.tokenPagination = tokenPagination;
    }

    public String getVisitorId() {
        return visitorId;
    }

    public void setVisitorId(String visitorId) {
        this.visitorId = visitorId;
    }
}
