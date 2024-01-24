package youtube.api.resources;

import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.jaxrs.annotation.JacksonFeatures;
import org.springframework.stereotype.Component;
import youtube.api.model.ErrorDescription;
import youtube.api.model.ErrorResponse;
import youtube.api.model.SearchListResponse;
import youtube.api.model.SearchRequest;
import youtube.api.repository.YouTubeRepository;
import youtube.api.repository.YouTubeRepositoryImpl;
import youtube.api.util.BadPaginationException;
import youtube.api.util.Location;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.GenericEntity;
import javax.ws.rs.core.Response;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

@Component
@Path("/search")
public class YouTubeResource {
    private static YouTubeResource instance = null;
    private static List<String> countryCodes = Arrays.asList("AF","AX","AL","DZ","AS","AD","AO","AI","AQ","AG","AR","AM","AW","AU","AT","AZ","BS","BH","BD","BB","BY","BE","BZ","BJ","BM","BT","BO","BQ","BA","BW","BV","BR","IO","BN","BG","BF","BI","KH","CM","CA","CV","KY","CF","TD","CL","CN","CX","CC","CO","KM","CG","CD","CK","CR","CI","HR","CU","CW","CY","CZ","DK","DJ","DM","DO","EC","EG","SV","GQ","ER","EE","ET","FK","FO","FJ","FI","FR","GF","PF","TF","GA","GM","GE","DE","GH","GI","GR","GL","GD","GP","GU","GT","GG","GN","GW","GY","HT","HM","VA","HN","HK","HU","IS","IN","ID","IR","IQ","IE","IM","IL","IT","JM","JP","JE","JO","KZ","KE","KI","KP","KR","KW","KG","LA","LV","LB","LS","LR","LY","LI","LT","LU","MO","MK","MG","MW","MY","MV","ML","MT","MH","MQ","MR","MU","YT","MX","FM","MD","MC","MN","ME","MS","MA","MZ","MM","NA","NR","NP","NL","NC","NZ","NI","NE","NG","NU","NF","MP","NO","OM","PK","PW","PS","PA","PG","PY","PE","PH","PN","PL","PT","PR","QA","RE","RO","RU","RW","BL","SH","KN","LC","MF","PM","VC","WS","SM","ST","SA","SN","RS","SC","SL","SG","SX","SK","SI","SB","SO","ZA","GS","SS","ES","LK","SD","SR","SJ","SZ","SE","CH","SY","TW","TJ","TZ","TH","TL","TG","TK","TO","TT","TN","TR","TM","TC","TV","UG","UA","AE","GB","US","UM","UY","UZ","VU","VE","VN","VG","VI","WF","EH","YE","ZM","ZW");
    private static List<String> languageCodes = Arrays.asList("AB","AA","AF","AK","SQ","AM","AR","AN","HY","AS","AV","AE","AY","AZ","BM","BA","EU","BE","BN","BH","BI","BS","BR","BG","MY","CA","CH","CE","NY","ZH","CV","KW","CO","CR","HR","CS","DA","DV","NL","DZ","EN","EO","ET","EE","FO","FJ","FI","FR","FF","GL","KA","DE","EL","GN","GU","HT","HA","HE","HZ","HI","HO","HU","IA","ID","IE","GA","IG","IK","IO","IS","IT","IU","JA","JV","KL","KN","KR","KS","KK","KM","KI","RW","KY","KV","KG","KO","KU","KJ","LA","LB","LG","LI","LN","LO","LT","LU","LV","GV","MK","MG","MS","ML","MT","MI","MR","MH","MN","NA","NV","ND","NE","NG","NB","NN","NO","II","NR","OC","OJ","CU","OM","OR","OS","PA","PI","FA","PL","PS","PT","QU","RM","RN","RO","RU","SA","SC","SD","SE","SM","SG","SR","GD","SN","SI","SK","SL","SO","ST","ES","SU","SW","SS","SV","TA","TE","TG","TH","TI","BO","TK","TL","TN","TO","TR","TS","TT","TW","TY","UG","UK","UR","UZ","VE","VI","VO","WA","CY","WO","FY","XH","YI","YO","ZA","ZU");
    YouTubeRepository repository;

    private static final String BAD_REQUEST_UNKNOWN_PARAMETER = "Unknown parameter: ";
    private static final String BAD_REQUEST_MISSING_PART = "Required parameter: part";
    private static final String BAD_REQUEST_DEPENDENCIES_START_MESSAGE = "Incompatible parameters specified in the request: ";
    private static final String BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS = "The request contains an invalid combination of search filters and/or restrictions." +
            " Note that you must set the \u003ccode\u003etype\u003c/code\u003e parameter to \u003ccode\u003evideo\u003c/code\u003e if you set either the" +
            " \u003ccode\u003eforContentOwner\u003c/code\u003e or \u003ccode\u003eforMine\u003c/code\u003e parameters to \u003ccode\u003etrue\u003c/code\u003e." +
            " You must also set the \u003ccode\u003etype\u003c/code\u003e parameter to \u003ccode\u003evideo\u003c/code\u003e if you set a value for the" +
            " \u003ccode\u003eeventType\u003c/code\u003e, \u003ccode\u003evideoCaption\u003c/code\u003e, \u003ccode\u003evideoCategoryId\u003c/code\u003e," +
            " \u003ccode\u003evideoDefinition\u003c/code\u003e, \u003ccode\u003evideoDimension\u003c/code\u003e, \u003ccode\u003evideoDuration\u003c/code\u003e," +
            " \u003ccode\u003evideoEmbeddable\u003c/code\u003e, \u003ccode\u003evideoLicense\u003c/code\u003e, \u003ccode\u003evideoSyndicated\u003c/code\u003e," +
            " or \u003ccode\u003evideoType\u003c/code\u003e parameters.";
    private static final String BAD_REQUEST_INVALID_VALUE = "Invalid value '";
    private static final String BAD_REQUEST_INVALID_STRING_VALUE = "Invalid string value: '";
    private static final String BAD_REQUEST_INVALID_LOCATION = "Invalid location.";
    private static final String BAD_REQUEST_INVALID_REGION_CODE = "The \u003ccode\u003eregionCode\u003c/code\u003e parameter specifies an invalid region code.";
    private static final String BAD_REQUEST_INVALID_RELEVANCE_LANGUAGE = "Invalid relevance language.";
    private static final String BAD_REQUEST_INVALID_PAGE_TOKEN = "The request specifies an invalid page token.";

    private static final String ERROR_DOMAIN_YOUTUBE_SEARCH = "youtube.search";
    private static final String ERROR_DOMAIN_YOUTUBE_PARAMETER = "youtube.parameter";
    private static final String ERROR_DOMAIN_YOUTUBE_PART = "youtube.part";
    private static final String ERROR_DOMAIN_GLOBAL = "global";


    private YouTubeResource() throws ParseException {
        this.repository = YouTubeRepositoryImpl.getInstance();
    }

    public static YouTubeResource getInstance() throws ParseException {
        if(instance == null) {
            instance = new YouTubeResource();
        }
        return instance;
    }

    @GET
    @Produces("application/json")
    @JacksonFeatures(serializationDisable = {SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, SerializationFeature.FAIL_ON_EMPTY_BEANS})
    public Response list(@BeanParam SearchRequest searchRequest, @Context HttpServletRequest request) {

        if(searchRequest.getPart() == null) {
            throw new BadRequestException(BAD_REQUEST_MISSING_PART, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_MISSING_PART, "part")).build());
        }

        String partType = validatePart(searchRequest.getPart());

        validateDependencies(searchRequest.getForContentOwner(), searchRequest.getForDeveloper(), searchRequest.getForMine(), searchRequest.getRelatedToVideoId());

        if((searchRequest.getForContentOwner() != null && searchRequest.getForContentOwner() && searchRequest.getOnBehalfOfContentOwner() != null && !searchRequest.getOnBehalfOfContentOwner().isEmpty()) || (searchRequest.getForMine() != null && searchRequest.getForMine())) {
            validateForContentOwnerOrForMineDependencies(searchRequest);
        } else if(searchRequest.getForContentOwner() != null && searchRequest.getForContentOwner()) {
            throw new BadRequestException(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, "")).build());
        } else if(searchRequest.getRelatedToVideoId() != null && !searchRequest.getRelatedToVideoId().isEmpty() && (!"video".equals(searchRequest.getType()))) {
            throw new BadRequestException(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, "")).build());
        } else {
            validateTypeDependencies(searchRequest);
        }

        validateChannelType(searchRequest.getChannelType());

        validateEventType(searchRequest.getEventType());

        Location locationAndLocationRadius = validateLocationAndLocationRadius(searchRequest.getLocation(), searchRequest.getLocationRadius());

        validateMaxResults(searchRequest);

        validateOrder(searchRequest);

        Date[] dates = validateDates(searchRequest.getPublishedBeforeStr(), searchRequest.getPublishedAfterStr());

        validateRegionCode(searchRequest.getRegionCode());

        validateRelevanceLanguage(searchRequest.getRelevanceLanguage());

        validateSafeSearch(searchRequest.getSafeSearch());

        validateVideoCaption(searchRequest.getVideoCaption());

        validateVideoDefinition(searchRequest.getVideoDefinition());

        validateVideoDimension(searchRequest.getVideoDimension());

        validateVideoDuration(searchRequest.getVideoDuration());

        validateVideoEmbeddable(searchRequest.getVideoEmbeddable());

        validateVideoLicense(searchRequest.getVideoLicense());

        validateVideoSyndicated(searchRequest.getVideoSyndicated());

        validateVideoType(searchRequest.getVideoType());

        byte typeSearch = searchRequest.getType() == null ||searchRequest.getType().isEmpty()? 0b111 : processTypeSearch(searchRequest.getType());

        SearchListResponse search;
        try {
            search = repository.searchList(searchRequest, partType, locationAndLocationRadius, dates, typeSearch, request.getLocale());
        } catch (BadPaginationException e) {
            throw new BadRequestException(BAD_REQUEST_INVALID_PAGE_TOKEN, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_PAGE_TOKEN, "pageToken")).build());
        }

        GenericEntity<SearchListResponse> entity = new GenericEntity<SearchListResponse>(search){};

        return Response.ok(entity).build();
    }

    private String validatePart(String part) {
        boolean snippet = false;
        boolean id = false;
        String[] sp = part.split(",");
        for(String s : sp) {
            String sTrim = s.trim();
            if(sTrim.equals("snippet")) {
                snippet = true;
            } else if(sTrim.equals("id")) {
                id = true;
            } else {
                throw new BadRequestException(part, Response.status(400).entity(generateBadRequestResponse(part, "part")).build());
            }
        }

        if(snippet && id) {
            return "both";
        }

        return snippet? "snippet" : "id";
    }

    private void validateDependencies(Boolean forContentOwner, Boolean forDeveloper, Boolean forMine, String relatedToVideoId) {
        if ((forContentOwner != null || forDeveloper != null || forMine != null) && relatedToVideoId != null && !relatedToVideoId.isEmpty()) {
            String message = BAD_REQUEST_DEPENDENCIES_START_MESSAGE
                    + (forContentOwner != null ? "forContentOwner, " : "")
                    + (forDeveloper != null ? "forDeveloper, " : "")
                    + (forMine != null ? "forMine, " : "")
                    + "relatedTo";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "")).build());
        }

        validateDependencies(forContentOwner, forDeveloper, forMine);
    }

    private void validateDependencies(Boolean forContentOwner, Boolean forDeveloper, Boolean forMine) {
        if((forContentOwner != null || forDeveloper != null) && forMine != null) {
            String message = BAD_REQUEST_DEPENDENCIES_START_MESSAGE
                    + (forContentOwner != null ? "forContentOwner, " : "")
                    + (forDeveloper != null ? "forDeveloper, " : "")
                    + "forMine";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "")).build());
        }

        validateDependencies(forContentOwner, forDeveloper);
    }

    private void validateDependencies(Boolean forContentOwner, Boolean forDeveloper) {
        if(forContentOwner != null && forDeveloper != null) {
            String message = BAD_REQUEST_DEPENDENCIES_START_MESSAGE + "forContentOwner, forDeveloper";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "")).build());
        }
    }

    private void validateForContentOwnerOrForMineDependencies(SearchRequest searchRequest) {
        if(searchRequest.getType() == null || !searchRequest.getType().equals("video") || searchRequest.getVideoDefinition() != null || searchRequest.getVideoDimension() != null || searchRequest.getVideoDuration() != null || searchRequest.getVideoLicense() != null ||
                searchRequest.getVideoEmbeddable() != null || searchRequest.getVideoSyndicated() != null || searchRequest.getVideoType() != null) {
            throw new BadRequestException(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, "")).build());
        }
    }

    private void validateTypeDependencies(SearchRequest searchRequest) {
        if(((searchRequest.getLocation() != null || searchRequest.getEventType() != null || searchRequest.getVideoCaption() != null || searchRequest.getVideoCategoryId() != null || searchRequest.getVideoDefinition() != null || searchRequest.getVideoDimension() != null || searchRequest.getVideoDuration() != null ||
                searchRequest.getVideoEmbeddable() != null || searchRequest.getVideoLicense() != null || searchRequest.getVideoSyndicated() != null || searchRequest.getVideoType() != null) && (searchRequest.getType() == null || !searchRequest.getType().equals("video"))) ||
                (searchRequest.getChannelType() != null && (searchRequest.getType() == null || !searchRequest.getType().equals("channel")))) {
            throw new BadRequestException(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS, "")).build());
        }
    }

    private void validateChannelType(String channelType) {
        if(channelType != null && !channelType.equals("any") && !channelType.equals("show")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + channelType + "'. Allowed values: [any, show]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "channelType")).build());
        }
    }

    private void validateEventType(String eventType) {
        if(eventType != null && !eventType.equals("completed") && !eventType.equals("live") && !eventType.equals("upcoming")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + eventType + "'. Allowed values: [completed, live, upcoming]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "eventType")).build());
        }
    }

    private Location validateLocationAndLocationRadius(String location, String locationRadius) {
        String parameter = "location";
        Location result = new Location();
        if(location != null && !location.isEmpty() && locationRadius != null && !locationRadius.isEmpty() &&
                location.matches("^[-+]?([1-8]?\\d(\\.\\d+)?|90(\\.0+)?),\\s*[-+]?(180(\\.0+)?|((1[0-7]\\d)|([1-9]?\\d))(\\.\\d+)?)$") &&
                locationRadius.matches("(^[1-9](\\d+)?(\\.\\d+)?|0\\.\\d+)(m|km|ft|mi)$")) {
            String[] sp = locationRadius.split("(?<=[a-z])(?=\\d)|(?<=\\d)(?=[a-z])");
            double radius = Double.parseDouble(sp[0]);
            switch (sp[1]) {
                case "m":
                    radius /= 1000.;
                    break;
                case "km":
                    break;
                case "ft":
                    radius /= 3280.84;
                    break;
                case "mi":
                    radius *= 1.60934;
                    break;
                default:
                    throw new BadRequestException(BAD_REQUEST_INVALID_LOCATION, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_LOCATION, parameter)).build());
            }
            if(radius > 1000.) {
                throw new BadRequestException(BAD_REQUEST_INVALID_LOCATION, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_LOCATION, parameter)).build());
            }

            String[] latLon = location.split(",");
            result.setLatitude(Double.parseDouble(latLon[0].trim()));
            result.setLongitude(Double.parseDouble(latLon[1].trim()));
            result.setRadius(radius*1000);

        } else if((location != null && !location.isEmpty()) || (locationRadius != null && !locationRadius.isEmpty())) {
            throw new BadRequestException(BAD_REQUEST_INVALID_LOCATION, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_LOCATION, parameter)).build());
        }

        return result;
    }

    private void validateMaxResults(SearchRequest searchRequest) {
        if(searchRequest.getMaxResults() != null && (searchRequest.getMaxResults() < 0 || searchRequest.getMaxResults() > 50)) {
            String message = BAD_REQUEST_INVALID_VALUE + searchRequest.getMaxResults() + "'. Values must be within the range: [0, 50]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "maxResults")).build());
        } else if(searchRequest.getMaxResults() == null) {
            searchRequest.setMaxResults(5);
        }
    }

    private void validateOrder(SearchRequest searchRequest) {
        if(searchRequest.getOrder() != null && !searchRequest.getOrder().equals("date") && !searchRequest.getOrder().equals("rating") && !searchRequest.getOrder().equals("relevance") && !searchRequest.getOrder().equals("title") &&
                !searchRequest.getOrder().equals("videoCount") && !searchRequest.getOrder().equals("viewCount")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + searchRequest.getOrder() + "'. Allowed values: [date, rating, relevance, title, videoCount, viewCount]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "searchRequest.getOrder()")).build());
        } else if(searchRequest.getOrder() == null) {
            searchRequest.setOrder("relevance");
        }
    }

    private Date[] validateDates(String publishedBeforeStr, String publishedAfterStr) {
        Date[] dates = new Date[2];
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
        dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));

        try {
            if(publishedBeforeStr != null) {
                dates[0] = dateFormat.parse(publishedBeforeStr);
            }
            if(publishedAfterStr != null) {
                dates[1] = dateFormat.parse(publishedAfterStr);
            }
        } catch (ParseException e) {
            String message = "Bad Request";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "")).build());
        }

        return dates;
    }

    private void validateRegionCode(String regionCode) {
        if(regionCode != null && (!regionCode.matches("^[A-Za-z]{2}$") || !countryCodes.contains(regionCode.toUpperCase()))) {
            throw new BadRequestException(BAD_REQUEST_INVALID_REGION_CODE, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_REGION_CODE, "regionCode")).build());
        }
    }

    private void validateRelevanceLanguage(String relevanceLanguage) {
        if(relevanceLanguage != null && (!relevanceLanguage.matches("^[A-Za-z]{2}$") || !languageCodes.contains(relevanceLanguage.toUpperCase()))) {
            throw new BadRequestException(BAD_REQUEST_INVALID_RELEVANCE_LANGUAGE, Response.status(400).entity(generateBadRequestResponse(BAD_REQUEST_INVALID_RELEVANCE_LANGUAGE, "relevanceLanguage")).build());
        }

    }

    private void validateSafeSearch(String safeSearch) {
        if(safeSearch != null && !safeSearch.equals("moderate") && !safeSearch.equals("strict") && !safeSearch.equals("none")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + safeSearch + "'. Allowed values: [none, moderate, strict]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "safeSearch")).build());
        }
    }

    private void validateVideoCaption(String videoCaption) {
        if(videoCaption != null && !videoCaption.equals("any") && !videoCaption.equals("closedCaption") && !videoCaption.equals("none")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoCaption + "'. Allowed values: [any, closedCaption, none]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoCaption")).build());
        }
    }

    private void validateVideoDefinition(String videoDefinition) {
        if(videoDefinition != null && !videoDefinition.equals("any") && !videoDefinition.equals("high") && !videoDefinition.equals("standard")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoDefinition + "'. Allowed values: [any, high, standard]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoDefinition")).build());
        }
    }

    private void validateVideoDimension(String videoDimension) {
        if(videoDimension != null && !videoDimension.equals("2d") && !videoDimension.equals("3d") && !videoDimension.equals("any")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoDimension + "'. Allowed values: [2d, 3d, any]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoDimension")).build());
        }
    }

    private void validateVideoDuration(String videoDuration) {
        if(videoDuration != null && !videoDuration.equals("any") && !videoDuration.equals("long") && !videoDuration.equals("medium") && !videoDuration.equals("short")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoDuration + "'. Allowed values: [any, long, medium, short]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoDuration")).build());
        }
    }

    private void validateVideoEmbeddable(String videoEmbeddable) {
        if(videoEmbeddable != null && !videoEmbeddable.equals("any") && !videoEmbeddable.equals("true")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoEmbeddable + "'. Allowed values: [any, true]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoEmbeddable")).build());
        }
    }

    private void validateVideoLicense(String videoLicense) {
        if(videoLicense != null && !videoLicense.equals("any") && !videoLicense.equals("creativeCommon") && !videoLicense.equals("youtube")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoLicense + "'. Allowed values: [any, creativeCommon, youtube]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoLicense")).build());
        }
    }

    private void validateVideoSyndicated(String videoSyndicated) {
        if(videoSyndicated != null && !videoSyndicated.equals("any") && !videoSyndicated.equals("true")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoSyndicated + "'. Allowed values: [any, true]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoSyndicated")).build());
        }
    }

    private void validateVideoType(String videoType) {
        if(videoType != null && !videoType.equals("any") && !videoType.equals("episode") && !videoType.equals("movie")) {
            String message = BAD_REQUEST_INVALID_STRING_VALUE + videoType + "'. Allowed values: [any, episode, movie]";
            throw new BadRequestException(message, Response.status(400).entity(generateBadRequestResponse(message, "videoType")).build());
        }
    }

    /**
     * This method ascertains the types of resources that must be retrieved. The number returned can be between 0 and 2^3-1; its binary representation
     * indicates the resources requested by the user: <br><br>
     *  <ul>
     *      <li>{@code 0=000} means no type specified, so the API will retrieve all types of resources (channel, playlist and video)</li>
     *      <li>{@code 1=001} means that the user demanded only 'video' type</li>
     *      <li>{@code 2=010} means that the user demanded only 'playlist' type</li>
     *      <li>{@code 3=011} means that the user demanded 'playlist' and 'video' types</li>
     *      <li>{@code 4=100} means that the user demanded only 'channel' type</li>
     *      <li>{@code 5=101} means that the user demanded 'channel' and 'video' types</li>
     *      <li>{@code 6=110} means that the user demanded 'channel' and 'playlist' types</li>
     *      <li>{@code 7=111} means that the user demanded all types of resources (channel, playlist and video)</li>
     *  </ul>
     *
     * @param type the parameter received in the request
     * @return the corresponding number
     */
    private byte processTypeSearch(String type) {
        boolean channel = false;
        boolean playlist = false;
        boolean video = false;
        String[] sp = type.split(",");
        for(String s : sp) {
            switch (s.trim()) {
                case "channel":
                    channel = true;
                    break;
                case "playlist":
                    playlist = true;
                    break;
                case "video":
                    video = true;
                    break;
                default:
                    break;
            }
        }

        return (byte) ((channel? 0b100 : 0b0) + (playlist? 0b010 : 0b0) + (video? 0b001 : 0b0));
    }

    private ErrorResponse generateBadRequestResponse(String message, String parameter) {
        ErrorResponse response = new ErrorResponse();

        ErrorDescription description = new ErrorDescription();
        description.setMessage(message);
        description.setLocationType("parameter");
        description.setLocation(parameter);

        if(message.equals(BAD_REQUEST_MISSING_PART)) {
            description.setDomain(ERROR_DOMAIN_GLOBAL);
            description.setReason("required");
        } else if(parameter.equals("part")) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_PART);
            description.setReason("unknownPart");
        } else if(message.startsWith(BAD_REQUEST_DEPENDENCIES_START_MESSAGE)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_PARAMETER);
            description.setReason("incompatibleParameters");
        } else if(message.equals(BAD_REQUEST_DEPENDENCIES_OPTIONAL_PARAMETERS)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_SEARCH);
            description.setReason("invalidSearchFilter");
        } else if(message.startsWith(BAD_REQUEST_INVALID_STRING_VALUE) || message.startsWith(BAD_REQUEST_INVALID_VALUE)) {
            description.setDomain(ERROR_DOMAIN_GLOBAL);
            description.setReason("invalidParameter");
        } else if(message.equals(BAD_REQUEST_INVALID_LOCATION)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_SEARCH);
            description.setReason("invalidLocation");
        } else if(message.equals(BAD_REQUEST_INVALID_REGION_CODE)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_PARAMETER);
            description.setReason("invalidRegionCode");
        } else if(message.equals(BAD_REQUEST_INVALID_RELEVANCE_LANGUAGE)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_SEARCH);
            description.setReason("invalidRelevanceLanguage");
        } else if(message.equals(BAD_REQUEST_INVALID_PAGE_TOKEN)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_PARAMETER);
            description.setReason("invalidPageToken");
        } else if(message.startsWith(BAD_REQUEST_UNKNOWN_PARAMETER)) {
            description.setDomain(ERROR_DOMAIN_YOUTUBE_SEARCH);
            description.setReason("unknownParameter");
        }

        response.setErrors(Collections.singletonList(description));
        response.setMessage(message);
        response.setStatus(400);
        return response;
    }

}
